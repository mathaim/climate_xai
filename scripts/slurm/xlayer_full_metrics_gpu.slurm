#!/bin/bash
#SBATCH --job-name=xlmetg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/xlmetg_%j.out
cd /home/euh7ys/climate_xai
export NMAX=56980
export REQUIRE_GPU=1
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/xlayer_full_metrics.py
