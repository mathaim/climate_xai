#!/bin/bash
#SBATCH --job-name=psae_L15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L15_%j.out
#SBATCH --error=logs/psae_L15_%j.err

# Train PlainSAE (TopK + AuxK) on GraphCast layer 15 activations.
# Activations on standard storage (slower reads).

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate climate_xai

cd ~/climate_xai

python -u -m src.training.train_plain_sae \
    --layer 15 \
    --data_dir /standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer15 \
    --output_dir /scratch/euh7ys/climate_xai/checkpoints/plain_layer15 \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2
