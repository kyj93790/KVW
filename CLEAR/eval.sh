#!/bin/bash
forget_ratio=05
retain_ratio=95
model_id=Qwen/Qwen2-VL-2B-Instruct
cache_path=checkpoints/KVW_05
data_folder="data/CLEAR"
gpu=0

forget_cls_folder=forget${forget_ratio}_perturbed
forget_gen_folder=forget${forget_ratio}+tofu
retain_cls_folder=retain_perturbed
retain_gen_folder=retain${retain_ratio}+tofu
realface_folder=real_faces
realworld_folder=real_world
shot_num="zero_shots"
output_folder=${cache_path}/${shot_num}

CUDA_VISIBLE_DEVICES=$gpu python eval.py \
	--model_id ${model_id} \
	--cache_path ${cache_path} \
	--eval_list "forget retain realface realworld" \
	--output_folder ${output_folder}/forget${forget_ratio} \
	--shot_num ${shot_num} \
	--data_folder ${data_folder} \
	--forget_cls_folder ${forget_cls_folder} \
	--forget_gen_folder ${forget_gen_folder} \
	--retain_cls_folder ${retain_cls_folder} \
	--retain_gen_folder ${retain_gen_folder} \
	--realface_folder ${realface_folder} \
	--realworld_folder ${realworld_folder}