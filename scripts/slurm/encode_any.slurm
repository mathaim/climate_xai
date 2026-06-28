#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/enc_%x_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.encode_fulltrack_any $SAE
echo ALL DONE
