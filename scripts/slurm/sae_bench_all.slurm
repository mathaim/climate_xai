#!/bin/bash
#SBATCH --job-name=saeball
#SBATCH --partition=gpu-a100-80
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/saeball_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000 REQUIRE_GPU=1
export SAES=matry_L0,matry_L8,matry_L15,plain_L0,plain_L8,plain_L15
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/sae_bench.py
