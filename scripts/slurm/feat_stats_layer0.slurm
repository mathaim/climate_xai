#!/bin/bash
#SBATCH --job-name=feat_stats_L0
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/feat_stats_L0_%j.out
#SBATCH --error=logs/feat_stats_L0_%j.err
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate py312
cd ~/climate_xai
python -u -m src.analysis.compute_feature_stats --layer 0 --data_dir /scratch/euh7ys/climate_xai/activations/layer00
