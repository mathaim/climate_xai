#!/bin/bash
#SBATCH --job-name=msae_L0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/msae_L0_%j.out
#SBATCH --error=logs/msae_L0_%j.err
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"
conda activate py312
cd ~/climate_xai
OUT=/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer00
mkdir -p "$OUT"
python -u -m src.training.train_matryoshka_sae \
    --layer 0 --d_model 512 --n_latents 4096 \
    --group_sizes 256 512 1024 2048 4096 --target_l0 32 --n_steps 300000 \
    --batch_size 4096 --lr 2e-4 \
    --data_dir /scratch/euh7ys/climate_xai/activations/layer00 \
    --output_dir "$OUT" --log_every 500 --save_every 10000 --metrics_every 5000 --topk_mode per_sample
echo "MatryoshkaSAE L0 done. Models in $OUT"
