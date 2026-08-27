#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/containment_%j.out
module load anaconda
conda activate climate_xai
cd ~/climate_xai
PYTHONPATH=. python src/analysis/ar_intensity/containment_metrics.py
