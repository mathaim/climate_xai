#!/bin/bash
#SBATCH --job-name=ar_nonar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_nonar_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.compute_baseline_nonar
echo NONAR DONE
