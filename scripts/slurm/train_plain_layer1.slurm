#!/bin/bash
#SBATCH --job-name=psae_L1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L1_%j.out
#SBATCH --error=logs/psae_L1_%j.err

# Train PlainSAE (TopK + AuxK) on GraphCast layer 0 activations.
# Activations on standard storage (976 files).
# Note: GraphCast layer 0 = "layer 1" in project naming.
# File prefix is layer0000_ so we pass --layer 0.

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

python -u -m src.training.train_plain_sae \
    --layer 0 \
    --data_dir /standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer00 \
    --output_dir /scratch/euh7ys/climate_xai/checkpoints/plain_layer1 \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2

# Copy final model to standard storage (scratch gets purged after 90 days)
STANDARD_DIR=/standard/AikyamLab/madelyn/GraphCast/PlainSAE/checkpoints_layer1
mkdir -p "$STANDARD_DIR"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer1/final_model.pt "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer1/config.json "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer1/training_log.jsonl "$STANDARD_DIR/"
echo "Final model copied to $STANDARD_DIR"
