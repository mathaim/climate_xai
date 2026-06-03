#!/bin/bash
#SBATCH --job-name=psae_L0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L0_%j.out
#SBATCH --error=logs/psae_L0_%j.err

# Train PlainSAE (TopK + AuxK) on GraphCast layer 0 activations.
# Reads activations from scratch (fast); writes checkpoints directly to
# standard (durable — survives the 90-day scratch purge). Checkpoint writes
# are small + infrequent, so standard's slow I/O does not affect training speed.

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

OUTPUT_DIR=/standard/AikyamLab/madelyn/GraphCast/PlainSAE/checkpoints_layer0
mkdir -p "$OUTPUT_DIR"

python -u -m src.training.train_plain_sae \
    --layer 0 \
    --data_dir /scratch/euh7ys/climate_xai/activations/layer00 \
    --output_dir "$OUTPUT_DIR" \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2

echo "PlainSAE L0 training complete. Checkpoints in $OUTPUT_DIR"
