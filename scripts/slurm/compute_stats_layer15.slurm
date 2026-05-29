#!/bin/bash
#SBATCH --job-name=stats_L15
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/stats_L15_%j.out
#SBATCH --error=logs/stats_L15_%j.err

# Compute per-feature min/max normalization stats for layer 15 activations.
# Reads from Research Standard (slower), CPU-only, no GPU needed.
# Stats saved as feature_min.npy and feature_max.npy inside the data dir.
#
# Must complete before training can start (train_matryoshka_layer15.slurm).
#
# Usage:
#   sbatch scripts/slurm/compute_stats_layer15.slurm

mkdir -p ~/climate_xai/logs

module load anaconda
conda activate climate_xai

cd ~/climate_xai

python -u -m src.analysis.compute_activation_stats \
    --data_dir /standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer15 \
    --layer 15
