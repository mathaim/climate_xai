#!/bin/bash
#SBATCH --job-name=findchild
#SBATCH --partition=standard
#SBATCH --time=00:20:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/euh7ys/climate_xai/logs/findchild_%j.out
cd /home/euh7ys/climate_xai
echo "PARENT=$PARENT GRP=$GRP"
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/find_children.py
