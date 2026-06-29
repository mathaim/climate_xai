#!/bin/bash
#SBATCH --job-name=gfoot
#SBATCH --partition=standard
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/gfoot_%j.out
cd /home/euh7ys/climate_xai
echo "CONCEPTS=$CONCEPTS  N=$GLOBAL_N  OUT=$FOOT_OUT"
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/global_footprint.py
