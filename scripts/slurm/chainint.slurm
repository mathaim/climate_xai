#!/bin/bash
#SBATCH --job-name=chainint
#SBATCH --partition=standard
#SBATCH --time=00:25:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/chainint_%j.out
cd /home/euh7ys/climate_xai
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/chain_intensity.py
