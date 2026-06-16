#!/bin/bash
#SBATCH --job-name=ar_saefeat
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ar_saefeat_%j.out
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate py312
cd ~/climate_xai && python -u -m src.analysis.ar_intensity.build_sae_features
