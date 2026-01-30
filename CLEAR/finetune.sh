CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--save_dir checkpoints/qwen2B_vanilla \
	--batch_size 4 \
	--lr 1e-4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--is_oracle False

CUDA_VISIBLE_DEVICES=0,1 python finetune.py \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--save_dir checkpoints/qwen2B_oracle_5 \
	--batch_size 4 \
	--lr 1e-4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--is_oracle True