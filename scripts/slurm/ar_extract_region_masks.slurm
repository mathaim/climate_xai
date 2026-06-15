#!/bin/bash
#SBATCH --job-name=ar_extract
#SBATCH --partition=standard
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ar_extract_%j.out
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate py312
cd ~/climate_xai && python -u -m src.analysis.ar_intensity.extract_region_masks
