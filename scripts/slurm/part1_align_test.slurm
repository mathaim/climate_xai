#!/bin/bash
#SBATCH --job-name=p1al
#SBATCH --partition=standard
#SBATCH --time=00:50:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/p1al_%j.out
cd /home/euh7ys/climate_xai
echo "TS=$TS REGION=$REGION CONCEPT=$CONCEPT"
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/part1_align_test.py
