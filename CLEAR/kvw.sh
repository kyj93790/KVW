CUDA_VISIBLE_DEVICES=1 python3 -m baselines.KVW \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--forget_ratio 05 \
	--batch_size 1 \
	--num_epochs 1 \
	--phase weakening \
	--data_folder data/CLEAR \
	--save_dir checkpoints/KVW_05