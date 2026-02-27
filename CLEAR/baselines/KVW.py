import sys
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('.'))
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset
import torch
import json
import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, MllamaForConditionalGeneration, AutoTokenizer, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch
from torch.optim import AdamW

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _get_decoder_layers(m):
    if hasattr(m, "base_model"):
        m = m.base_model  # e.g. PeftModel → LlavaForConditionalGeneration or LlamaForCausalLM

    if hasattr(m, "language_model"):
        lm = m.language_model
    elif hasattr(m, "text_model"):
        lm = m.text_model
    elif hasattr(m, "model") and hasattr(m.model, "text_model"):
        lm = m.model.text_model
    else:
        lm = m

    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    
    if hasattr(lm, "layers"):
        return lm.layers

    raise ValueError(f"Unknown Decoder layers. type={type(m)}")

# Example usage:
def load_model_and_processor(args):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        model = MllamaForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
        )

        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    elif "qwen2.5" in args.model_id.lower():
        print("Loading Qwen2.5 model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir,
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"
    elif "qwen" in args.model_id.lower():
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.vanilla_dir, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


@torch.no_grad()
def compute_knowledge_coeffs_from_batch(model, batch, eps=1e-12):
    model.eval()

    layers = _get_decoder_layers(model)
    n_layers = len(layers)

    coeff_sums = [None for _ in range(n_layers)]
    token_counts = [0.0 for _ in range(n_layers)]
    current = {"attention_mask": None, "answer_mask": None}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            coeff = inputs[0]                      # [B, T, d_ff]
            coeff = coeff.to(torch.float32).abs()
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

            attn_mask = current["attention_mask"].to(coeff.device)  # [B, T]
            mask = attn_mask

            ans_mask = current["answer_mask"]
            ans_mask = ans_mask.to(coeff.device)                # [B, T] bool
            mask = mask.bool() & ans_mask                       # [B, T] bool

            mask_f = mask.unsqueeze(-1).to(coeff.dtype)             # [B, T, 1]
            coeff_masked = coeff * mask_f

            summed = coeff_masked.sum(dim=(0, 1)).detach().cpu()    # [d_ff]

            if coeff_sums[layer_idx] is None:
                coeff_sums[layer_idx] = summed
            else:
                coeff_sums[layer_idx] += summed

            token_counts[layer_idx] += mask.sum().item()
        return hook

    hooks = []
    for l, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(l)))

    labels = None
    if isinstance(batch, dict):
        attention_mask = batch["attention_mask"]
        labels = batch.get("labels", None)
    elif isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError(f"Batch length {len(batch)} < 2")
        attention_mask = batch[1]
        if len(batch) >= 4:
            labels = batch[3]
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    current["attention_mask"] = attention_mask
    cause_mask = torch.zeros_like(labels, dtype=torch.bool)
    cause_mask[:, :-1] = (labels[:, 1:] != -100)
    current["answer_mask"] = cause_mask

    # ---- forward ----
    _ = model(**batch)

    for h in hooks:
        h.remove()

    kc_list = []
    for l in range(n_layers):
        if coeff_sums[l] is None or token_counts[l] == 0:
            kc = torch.zeros(1)
        else:
            kc = coeff_sums[l] / max(token_counts[l], 1e-8)
            kc = torch.nan_to_num(kc, nan=0.0, posinf=0.0, neginf=0.0)
        kc_list.append(kc)

    return kc_list


@torch.no_grad()
def compute_knowledge_coeffs(
    model,
    dataloader,
    eps=1e-12,
):
    model.eval()
    layers = _get_decoder_layers(model)
    n_layers = len(layers)

    coeff_sums = [None for _ in range(n_layers)]
    token_counts = [0.0 for _ in range(n_layers)]
    current = {"attention_mask": None, "answer_mask": None}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            coeff = inputs[0]  # [B, T, m]
            coeff = coeff.to(torch.float32).abs()
            coeff = torch.nan_to_num(coeff, nan=0.0, posinf=0.0, neginf=0.0)

            attn_mask = current["attention_mask"].to(coeff.device)   # [B, T]
            mask = attn_mask.bool()

            ans_mask = current["answer_mask"]
            ans_mask = ans_mask.to(coeff.device)
            mask = mask & ans_mask

            mask_f = mask.unsqueeze(-1).to(coeff.dtype)              # [B, T, 1]
            summed = (coeff * mask_f).sum(dim=(0, 1)).detach().cpu() # [m]

            if coeff_sums[layer_idx] is None:
                coeff_sums[layer_idx] = summed
            else:
                coeff_sums[layer_idx] += summed

            token_counts[layer_idx] += mask.sum().item()
        return hook

    hooks = []
    for l, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_hook(make_hook(l)))

    def run_one_loader(loader, desc):
        for batch in tqdm(loader, desc=desc):
            labels = None

            if isinstance(batch, dict):
                attention_mask = batch["attention_mask"]
                labels = batch.get("labels", None)

            elif isinstance(batch, (list, tuple)):
                if len(batch) < 2:
                    raise ValueError(f"Batch length {len(batch)} < 2")

                attention_mask = batch[1]
                if len(batch) < 5:
                    raise ValueError(
                        f"Forget batch expects 5 items: "
                        f"(input_ids, attention_mask, pixel_values, noise_values, labels), got {len(batch)}"
                    )
                labels = batch[4]
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            current["attention_mask"] = attention_mask
            cause_mask = torch.zeros_like(labels, dtype=torch.bool)
            cause_mask[:, :-1] = (labels[:, 1:] != -100)
            current["answer_mask"] = cause_mask

            # forward
            _ = model(
                **batch
            )

    run_one_loader(dataloader, desc="Computing KC")
    for h in hooks:
        h.remove()

    kc_list = []
    for l in range(n_layers):
        if coeff_sums[l] is None or token_counts[l] == 0:
            kc = torch.zeros(1)
        else:
            kc = coeff_sums[l] / max(token_counts[l], eps)
            kc = torch.nan_to_num(kc, nan=0.0, posinf=0.0, neginf=0.0)
        kc_list.append(kc)

    return kc_list

def build_knowledge_ratio_gates(kc_f_list, kc_r_list, gamma, eps=1e-12):
    gates = []

    for kc_f, kc_r in zip(kc_f_list, kc_r_list):
        kc_f = kc_f.detach().cpu()
        kc_r = kc_r.detach().cpu()

        delta = torch.log(kc_f + eps) - torch.log(kc_r + eps)
        delta = torch.clamp(delta, min=0.0)
        g = torch.exp(-gamma * delta)   # delta=1 → gate≈0.37

        gates.append(g)

    return gates

def apply_weakening_to_model(model, gates, start_layer, end_layer, rescale=False):
    layers = _get_decoder_layers(model)
    assert len(layers) == len(gates)

    with torch.no_grad():
        for l, layer in enumerate(layers):
            if l < start_layer or l > end_layer:
                continue
            W = layer.mlp.down_proj.weight
            if rescale:
                W_old = W.clone()

            g = gates[l].to(W.device, dtype=W.dtype)  # [m]
            g = g.view(1, -1)
            W.mul_(g)   # in-place
            
            if rescale:
                old_norm = W_old.norm(p=2)
                new_norm = W.norm(p=2) + 1e-12
                scale = old_norm / new_norm
                W.mul_(scale)

######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)
    print(model)
    tokenizer = processor.tokenizer
    print("Tokenizer Length: ", len(tokenizer))

    # Resize token embeddings to match the tokenizer
    model.resize_token_embeddings(len(processor.tokenizer))
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
    
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    # Dataset and Dataloader setup
    forget_df=load_dataset(f"data/CLEAR/forget{args.forget_ratio:02}",split=f"train")#forget is the dataset that we want to forget
    retain_df=load_dataset(f"data/CLEAR/retain{100-args.forget_ratio}",split="train")#retain is the dataset that we want to preserve
    
    multimodal_forget_dataset = CLEAR_Dataset(data=forget_df,mode=CAPTION_MODE)
    multimodal_retain_dataset = CLEAR_Dataset(data=retain_df,mode=CAPTION_MODE)

    train_collate_function = train_collate_clear_ansonly

    device=model.device
    if processor:
        train_dataloader_forget = DataLoader(
            multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor,device, True)
        )
        train_dataloader_retain = DataLoader(
            multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_function(x, processor,device, True)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()
    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget) + len(train_dataloader_retain)) * args.num_epochs,
    )
    model, optimizer, train_dataloader_forget, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, lr_scheduler
    )
    n_iters = len(train_dataloader_forget)
    
    if args.phase == 'compute_kc_r':
        kc_r = compute_knowledge_coeffs(model, train_dataloader_retain)
        os.makedirs(f"kc",exist_ok=True)
        torch.save(kc_r, f"kc/kc_r_retain_{100 - args.forget_ratio:02}.pt")
        return

    kc_r = torch.load(f"kc/kc_r_retain_{100 - args.forget_ratio:02}.pt")
    for epoch in range(args.num_epochs):
        train_data_forget = enumerate(train_dataloader_forget)
        for iter in tqdm(range(0, n_iters)):
            try:
                _, forget_batch = next(train_data_forget)  # avoid StopIteration Error
                kc_f = compute_knowledge_coeffs_from_batch(
                model,
                forget_batch
                )
                gates, _ = build_knowledge_ratio_gates(kc_f, kc_r, args.gamma)
                apply_weakening_to_model(model, gates, args.start_layer, args.end_layer, False)
            except Exception as e:
                pass

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.save_dir, safe_serialization=True)
    print(f"Model saved to: {args.save_dir}")

if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_folder", type=str, default="../Data_split", help="Directory to save the model")
    parser.add_argument("--forget_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    parser.add_argument("--phase", type=str, default="weakening") # or compute_kc_r
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument("--start_layer", type=int, default=1)
    parser.add_argument("--end_layer", type=int, default=25)

    args = parser.parse_args()

    # Call main function
    main(args)
    if args.phase == 'weakening':
        with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
            json.dump(vars(args), f, indent=4)
