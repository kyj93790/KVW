import os
import json
import random
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, MllamaForConditionalGeneration,Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration
import pandas as pd
from io import BytesIO
from rouge_score import rouge_scorer
import argparse
import fnmatch
from datasets import load_dataset,load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_process.CLEAR_process import CLEAR_Dataset, CAPTION_MODE, RECOGNITION_MODE, train_collate_clear, NONE_MODE
import re
random.seed(42)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_classification(model, processor, data_path,with_options):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode, with_options={with_options} #########################################" )
    if "forget" in data_path:
        VQA_data=load_from_disk(data_path)
    elif "retain" in data_path:
        VQA_data=load_dataset(data_path,split="train")
    else:
        ValueError("Data path should contain forget or retain")
    print(VQA_data)
    correct_count,VQA_num = 0,0
    for idx,VQA_sample in enumerate(VQA_data):
        image=VQA_sample.get("image",None)
        question = VQA_sample.get("question", "What is the name of the person in the image?")
        answer = VQA_sample.get("name", "")
        options=VQA_sample.get("perturbed_names",[])
        options.insert(random.randint(0, len(options)), answer)
        if with_options:
            prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        else:
            prompt = question
            correct_answer=answer
        conversation = [
            {"role": "user","content": [{"type": "image"},{"type": "text", "text": prompt},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        
        with torch.no_grad():
            VQA_outputs = model.generate(**inputs,max_new_tokens=50, do_sample=False)
        
        out_wo_prompt = VQA_outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n","**************\n",generated_text,"\n**************")

        if not with_options: # answer in response is okay
            answer = re.sub(r'[^a-zA-Z0-9]', '', answer)
            if answer.lower() in assistant_response.lower():
                print("Correct Answer!")
                correct_count+=1
            else:
                print(f"Wrong Answer! ${assistant_response}$ doesn't include ${answer}$")
        else: # string matching
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
            modified_answer = re.sub(r'[^a-zA-Z0-9]', '', answer)
            if predicted_answer==correct_answer or modified_answer.lower() in assistant_response.lower():
                print("Correct Answer!")
                correct_count+=1
            else:
                print(f"Wrong Answer! ${predicted_answer}$ != ${correct_answer}$. {answer}")
        print("##################################")
        VQA_num+=1
    
    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}


def formulate_prompt_with_options(question, options, answer):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (list): The options for the question (e.g., ["Option A", "Option B"]).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{chr(ord('A')+i)}. {value}" for i,value in enumerate(options)])
    gt=chr(ord('A')+options.index(answer))
    prompt = f"{question}\n{options_str}\n"
    return prompt, gt

def eval_classification_real(model, processor, data_path):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {data_path} Mode #########################################" )
    df=load_dataset(data_path,split="train")
    correct_count,VQA_num = 0,0
    for i, sample in enumerate(df):
        question = sample.get("question", "What is the name of the person in the image?")
        answer = sample.get("answer", "")
        options=sample.get("options",[])
        options.insert(random.randint(0, len(options)), answer)
        image=sample.get("image",None)
        prompt, correct_answer = formulate_prompt_with_options(question, options, answer)
        conversation = [
            {"role": "user","content": [{"type": "image"},{"type": "text", "text": prompt},],},
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors='pt').to(model.device, torch.float16)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        out_wo_prompt = outputs[ : , inputs.input_ids.shape[-1] : ]
        generated_text=processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True)
        assistant_response = re.sub(r'[^a-zA-Z0-9]', '', generated_text)
        print("Generated text is : \n","**************\n",generated_text,"\n**************")
        predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() else None
        if predicted_answer==correct_answer:
            print("Correct Answer!")
            correct_count+=1
        else:
            print(f"Wrong Answer! ${assistant_response}$ != ${correct_answer}$. {answer}")
        VQA_num+=1
        print("##################################")
    
    print(f"VQA Correct Count: {correct_count}/{VQA_num}")
    print(f"VQA Accuracy: {correct_count/VQA_num}")
    print("################################## Classification Task Ends ##############################################")
    return {"VQA Accuracy": correct_count/VQA_num}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--forget_cls_folder', type=str, required=True, help='Path to the forget cls folder.')
    parser.add_argument('--forget_gen_folder', type=str, required=True, help='Path to the forget gen folder.')
    parser.add_argument('--retain_gen_folder', type=str, required=True, help='Path to the retain cls folder.')
    parser.add_argument('--retain_cls_folder', type=str, required=True, help='Path to the retain gen folder.')
    parser.add_argument('--realface_folder', type=str, required=True, help='Path to real person folder.')
    parser.add_argument('--realworld_folder', type=str, required=True, help='Path to real world folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder.')
    parser.add_argument('--eval_list', type=str, required=True, help='Spilts waited to eval')
    parser.add_argument('--shot_num', type=str, required=True, help='Shot nums for ICL')
    return parser.parse_args()

def main():
    args = parse_arguments()
    global few_shots_num
    if "zero" in args.shot_num.lower():
        few_shots_num=0
    else:
        few_shots_num=1

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()

    if "llava" in args.model_id.lower():
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    elif "llama" in args.model_id.lower():
        print("Loading LLAMA model...")
        model = MllamaForConditionalGeneration.from_pretrained(
            args.cache_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    elif "qwen2.5" in args.model_id.lower():
        print("Loading Qwen model...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.cache_path,
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True,
            local_files_only=True,
            attn_implementation="sdpa",
        )
    elif "qwen" in args.model_id.lower():
        print("Loading Qwen model...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.cache_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            local_files_only=True,
            attn_implementation="sdpa",
        )


    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    results_data={}
    if "forget" in args.eval_list:
        print("### Evaluating Forget Set ###")
        forget_classification_result = eval_classification(model=model, processor=processor, data_path=f"{args.data_folder}/{args.forget_cls_folder}_1",with_options=False)

        results_data["Forget Set Results 1"]={
            "classification": forget_classification_result,
        }
        
    if "forget" in args.eval_list:
        print("### Evaluating Forget Set ###")
        forget_classification_result = eval_classification(model=model, processor=processor, data_path=f"{args.data_folder}/{args.forget_cls_folder}_2",with_options=False)

        results_data["Forget Set Results 2"]={
            "classification": forget_classification_result,
        }

    if "retain" in args.eval_list:
        print("### Evaluating Retain Shared Set ###")
        print(f"{args.data_folder}/{args.retain_cls_folder}")
        retain_classification_result = eval_classification(model=model, processor=processor, data_path=f"{args.data_folder}/{args.retain_cls_folder}", wo_options=True)

        results_data["Retain Set Results"]= {
            "classification": retain_classification_result,
            # "generation": retain_generation_result
        }
    
    if "realface" in args.eval_list:
        print("### Evaluating Real Face Set ###")

        realface_classification_result = eval_classification_real(model=model, processor=processor, data_path=f"{args.data_folder}/{args.realface_folder}")

        print("Real Face Results:")
        print(realface_classification_result)
        results_data['Real Face Results']={
            "classification": realface_classification_result
        }
    
    if "realworld" in args.eval_list:
        print("### Evaluating Real World Set ###")

        realworld_classification_result = eval_classification_real(model=model, processor=processor, data_path=f"{args.data_folder}/{args.realworld_folder}")

        print("Real World Results:")
        print(realworld_classification_result)
        results_data['Real World Results']={
            "classification": realworld_classification_result
        }

    output_file = f'{args.output_folder}/final_evaluation_results.json'

    os.makedirs(args.output_folder, exist_ok=True)
    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")

    with open(f'{args.output_folder}/evalconfig.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()