#!/bin/bash
#SBATCH --job-name=ct_pool
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ct_pool_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.encode_fulltrack_pool
echo ALL DONE
