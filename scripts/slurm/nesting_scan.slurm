#!/bin/bash
#SBATCH --job-name=nscan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/nscan_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000
export REQUIRE_GPU=1
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/nesting_scan.py
