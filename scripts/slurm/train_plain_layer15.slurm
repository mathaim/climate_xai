#!/bin/bash
#SBATCH --job-name=psae_L15
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L15_%j.out
#SBATCH --error=logs/psae_L15_%j.err

# Train PlainSAE (TopK + AuxK) on GraphCast Layer 15 activations.
# Reads activations from /project (Ceph); writes checkpoints directly to the
# durable /project train tree (no scratch copy needed).

mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"
conda activate py312
cd ~/climate_xai

OUT=/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer15
mkdir -p "$OUT"

python -u -m src.training.train_plain_sae \
    --layer 15 \
    --data_dir /project/AikyamLab/madelyn/GraphCast/activations/Layer15 \
    --output_dir "$OUT" \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2

echo "PlainSAE L15 done. Models in $OUT"
