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
# Timestamps discovered from era5_inputs filenames.
# Output → /scratch (each file ~80MB, total ~4.4TB for 56,982 timesteps).
#
# ⚠️  Verify scratch has enough free space before submitting:
#     df -h /scratch/euh7ys
#
# Submit:
#     sbatch scripts/slurm/extract_layer0.slurm

mkdir -p ~/climate_xai/logs

# Use the SAME working environment that successfully extracted L8 and L15:
# the graphcast_sae_env venv (CUDA12 JAX + the custom graphcast fork).
# Do NOT use the climate_xai conda env — it has CPU-only JAX and an
# incompatible xarray that breaks extraction.
module load cuda/12.8.0
source /home/euh7ys/ondemand/Research/GraphCast/graphcast-interpretability/graphcast_sae_env/bin/activate
export PYTHONNOUSERSITE=1

cd ~/climate_xai

# GraphCast checkpoint
export GRAPHCAST_CHECKPOINT="/scratch/euh7ys/climate_xai/graphcast_checkpoints/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"

# Each array task processes 5000 timesteps starting at its offset
CHUNK=5000
START_IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

echo "Array task $SLURM_ARRAY_TASK_ID: start_idx=$START_IDX count=$CHUNK"

python -u -m src.extraction.extract_activations \
    --layer 0 \
    --era5_dir /scratch/euh7ys/climate_xai/era5_inputs \
    --start_idx $START_IDX \
    --count $CHUNK \
    --base_output_dir /scratch/euh7ys/climate_xai/activations
