#!/bin/bash
#SBATCH --job-name=psae_L8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L8_%j.out
#SBATCH --error=logs/psae_L8_%j.err

# Train PlainSAE (TopK + AuxK) on GraphCast layer 8 activations.
# Activations on scratch (fast reads).

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

python -u -m src.training.train_plain_sae \
    --layer 8 \
    --data_dir /scratch/euh7ys/activations_layer08_train \
    --output_dir /scratch/euh7ys/climate_xai/checkpoints/plain_layer8 \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2

# Copy final model to standard storage (scratch gets purged after 90 days)
STANDARD_DIR=/standard/AikyamLab/madelyn/GraphCast/PlainSAE/checkpoints_layer8
mkdir -p "$STANDARD_DIR"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer8/final_model.pt "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer8/config.json "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer8/training_log.jsonl "$STANDARD_DIR/"
echo "Final model copied to $STANDARD_DIR"
