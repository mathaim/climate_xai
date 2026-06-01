#!/bin/bash
#SBATCH --job-name=ext_L0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-11
#SBATCH --output=logs/ext_L0_%A_%a.out
#SBATCH --error=logs/ext_L0_%A_%a.err

# Extract GraphCast Layer 0 activations (processor step 0, 1st round).
# Uses SLURM array: 12 tasks × 5000 timesteps each = 60,000 capacity.
# Output → /scratch (each file ~80MB, total ~4.4TB for 56,982 timesteps).
#
# ⚠️  Verify scratch has enough free space before submitting:
#     df -h /scratch/euh7ys
#
# Submit:
#     sbatch scripts/slurm/extract_layer0.slurm

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

# Each array task processes 5000 timesteps starting at its offset
CHUNK=5000
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "Array task $SLURM_ARRAY_TASK_ID: start_idx=$START_IDX count=$CHUNK"

python -u -m src.extraction.extract_activations \
    --layer 0 \
    --timestamps_file ~/climate_xai/data/train_timestamps.txt \
    --start_idx $START_IDX \
    --count $CHUNK \
    --base_output_dir /scratch/euh7ys/climate_xai/activations
