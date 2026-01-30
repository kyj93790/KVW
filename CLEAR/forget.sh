CUDA_VISIBLE_DEVICES=0,1 python -m baselines.GA \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--data_folder data/CLEAR \
	--rank 4 \
	--save_dir checkpoints/GA_5

CUDA_VISIBLE_DEVICES=0,1 python -m baselines.GA_Diff \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--data_folder data/CLEAR \
	--rank 4 \
	--save_dir checkpoints/GD_5

CUDA_VISIBLE_DEVICES=0,1 python -m baselines.KL_Min \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--data_folder data/CLEAR \
	--rank 4 \
	--save_dir checkpoints/KL_5

CUDA_VISIBLE_DEVICES=0,1 python -m baselines.NPO \
	--model_id Qwen/Qwen2-VL-2B-Instruct \
	--vanilla_dir checkpoints/qwen2B_vanilla \
	--lr 1e-5 \
	--batch_size 4 \
	--num_epochs 1 \
	--forget_ratio 5 \
	--lcoef 1 \
	--beta 0.4 \
	--data_folder data/CLEAR \
	--rank 4 \
	--save_dir checkpoints/GD_5 \
	--oracle_model_id checkpoints/qwen2B_vanilla