#!/bin/bash
#SBATCH --job-name=ar_ivt
#SBATCH --partition=standard
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/ar_ivt_%j.out
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate base
cd ~/climate_xai && python -u -m src.analysis.ar_intensity.build_intensity
