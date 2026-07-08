#!/bin/bash
#SBATCH --job-name=pers
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/macro_persistence_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000
export REQUIRE_GPU=1
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
