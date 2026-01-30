# Experiments on CLEAR
Guide for Experiments on the CLEAR Benchmark

## 🔧 Environment Setup
1. Create Conda Environment
```
conda create -n clear python=3.10 -y
conda activate clear
```
2. Install System Dependencies
```
apt-get update -y
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get install -y tmux build-essential ninja-build
apt-get install -y cuda-toolkit-12-1
```
3. Configure CUDA Environment Variables
```
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

source ~/.bashrc
conda activate clear # Make sure to reactivate the clear environment!
```
4. Install Python Dependencies
```
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
apt-get install -y libopenmpi-dev openmpi-bin
```
## Download the Dataset
```
mkdir data # Current working directory: KVW/CLEAR
cd data
git clone https://huggingface.co/datasets/therem/CLEAR
```
If you wish to evaluate using the `2-fold cross-validation protocol` adopted in our work, download the additional dataset as follows:
```
git clone https://huggingface.co/datasets/yejinkim/clear-two-fold-val
cd clear-two-fold-val
mv * ../CLEAR
```

## Getting Vanilla Model & Oracle Model
```
bash finetune.sh
```
## Running Baselines
### For GA, GD, KL, NPO
```
bash forget.sh
```
※ lcoef: loss coefficient for balancing the forget and retain losses

### For MMU
First, generate the gradient map as follows:
```
CUDA_VISIBLE_DEVICES=0 python data_process/gen_mask.py
```
Second, run the selective unlearning process:
```
python MMU.py --model_id Qwen/Qwen2-VL-2B-Instruct --forget_ratio 05 --save_dir path_to_save_dir --batch_size 4 --lr 1e-5 --num_epochs 1 --grad_mask_path "path_to/language_mask.pt"
```
## Running Knowledge Vector Weakening (KVW)
First, precompute KC_r
```
bash compute_kc_r.sh
```
Second, run the knowledge vector weakening procss:
```
bash kvw.sh
```
## Evaluation
```
bash eval.sh
```
