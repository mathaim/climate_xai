#!/bin/bash
#SBATCH --job-name=msae_L15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/msae_L15_%j.out
#SBATCH --error=logs/msae_L15_%j.err

# Train MatryoshkaSAE (BatchTopK) on GraphCast Layer 15 activations.
# Reads from /project (Ceph); writes checkpoints to the durable /project tree.
# Stats (feature_min/max) already present in the activations dir.

mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"
conda activate py312
cd ~/climate_xai

OUT=/project/AikyamLab/madelyn/GraphCast/train/MatryoshkaSAE/Layer15
mkdir -p "$OUT"

python -u -m src.training.train_matryoshka_sae \
    --layer 15 \
    --d_model 512 \
    --n_latents 4096 \
    --group_sizes 256 512 1024 2048 4096 \
    --target_l0 32 \
    --n_steps 300000 \
    --batch_size 4096 \
    --lr 3e-2 \
    --data_dir /project/AikyamLab/madelyn/GraphCast/activations/Layer15 \
    --output_dir "$OUT" \
    --log_every 500 \
    --save_every 10000 \
    --metrics_every 5000

echo "MatryoshkaSAE L15 done. Models in $OUT"
