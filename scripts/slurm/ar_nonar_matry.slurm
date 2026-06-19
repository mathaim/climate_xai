#!/bin/bash
#SBATCH --job-name=ar_nonarM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=6:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_nonarM_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export NONAR_SAES=matry_L0,matry_L8,matry_L15
python -u -m src.analysis.ar_intensity.compute_baseline_nonar_layers
echo ALL DONE
