import sys
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
from transformers import LlavaForConditionalGeneration, AutoProcessor, get_scheduler, MllamaForConditionalGeneration,AutoTokenizer,Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE,train_collate_clear_ansonly
from accelerate import Accelerator
import torch
import torch.nn.functional as F
from torch.optim import AdamW

def find_all_linear_names(model):
    print(model)
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["embeddings","embed_tokens","patch_embed"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else ".".join(names[-2:]))

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    # if "qwen" in str(model).lower():
    #     lora_module_names.remove('proj')
    return list(lora_module_names)

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
        # LoRA configuration
        lora_config = LoraConfig(
            r=args.rank, #32
            lora_alpha=8, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    
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
        # LoRA configuration
        lora_config = LoraConfig(
            r=16, #32
            lora_alpha=16, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

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
        lora_config = LoraConfig(
            r=args.rank, #32
            lora_alpha=8, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )
        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
        # LoRA configuration
        lora_config = LoraConfig(
            r=args.rank, #32
            lora_alpha=8, #8
            lora_dropout=0.05,
            # target_modules=["q_proj", "v_proj"],
            target_modules=find_all_linear_names(model),
            init_lora_weights="gaussian",
        )

        print("getting peft model")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        processor = AutoProcessor.from_pretrained(args.model_id)
        processor.tokenizer.padding_side = "right"  # Ensure right padding
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor



######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args)

    if "llava" in args.model_id:
        # Load LLAVA model and processor
        print("Loading Oracle LLAVA model...")
        oracle_model = LlavaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "llama" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle llama model...")
        oracle_model = MllamaForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "qwen2.5" in args.model_id.lower():
        print("Loading Oracle qwen3B model...")
        oracle_model = model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif "qwen" in args.model_id.lower():
        # Load LLAVA Next model and processor
        print("Loading Oracle qwen model...")
        oracle_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.oracle_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )


    print("Processor Tokenizer Length: ", len(processor.tokenizer)) #128257
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))


    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

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
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader_forget)) * args.num_epochs,
    )

    # Prepare with accelerator
    model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_forget, train_dataloader_retain, lr_scheduler
    )

    len_retain, len_forget = len(train_dataloader_retain), len(train_dataloader_forget)
    n_iters = len(train_dataloader_forget)

    for epoch in range(args.num_epochs):
        train_data_forget=enumerate(train_dataloader_forget)
        train_data_retain=enumerate(train_dataloader_retain)
        model.train()
        total_loss = 0

        if args.gradient_accumulation:
            pass
        else:
            for iter in tqdm(range(0, n_iters)):
                # Forward pass with the current model to get the loss
                _, forget_batch = next(train_data_forget)  # avoid StopIteration Error
                forget_outputs = model(**forget_batch)
                current_loss = forget_outputs.loss

                # Forward pass with the oracle model to get the oracle loss
                with torch.no_grad():
                    oracle_outputs = oracle_model(**forget_batch)
                    oracle_loss = oracle_outputs.loss

                # Compute neg_log_ratios and NPO loss
                neg_log_ratios = current_loss - oracle_loss
                loss = -F.logsigmoid(args.beta * neg_log_ratios).mean() * 2 / args.beta
                
                _, retain_batch = next(train_data_retain)
                retain_outputs = model(**retain_batch)
                loss_retain = retain_outputs.loss
                loss += args.lcoef * loss_retain

                # Backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                total_loss += loss.item()

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if isinstance(model, PeftModel):
        unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--oracle_model_id", type=str, required=True, help="Oracle model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--forget_ratio", type=int, default=5, help="Directory to save the model")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--data_folder", type=str, default="./data", help="Data folder")
    parser.add_argument("--lcoef", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=4)

    args = parser.parse_args()
    main(args)
    with open(f"{args.save_dir}/trainer_config.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)
