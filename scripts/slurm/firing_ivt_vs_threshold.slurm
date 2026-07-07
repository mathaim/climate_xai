#!/bin/bash
#SBATCH --job-name=fivt
#SBATCH --partition=standard
#SBATCH --time=00:50:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/fivt_%j.out
cd /home/euh7ys/climate_xai
echo "TS=$TS REGION=$REGION CONCEPT=$CONCEPT"
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/firing_ivt_vs_threshold.py
