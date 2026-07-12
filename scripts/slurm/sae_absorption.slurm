#!/bin/bash
#SBATCH --job-name=absorb
#SBATCH --partition=gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/absorb_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000 REQUIRE_GPU=1
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/sae_absorption.py
