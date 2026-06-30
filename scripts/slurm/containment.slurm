#!/bin/bash
#SBATCH --job-name=contain
#SBATCH --partition=standard
#SBATCH --time=00:25:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/contain_%j.out
cd /home/euh7ys/climate_xai
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/find_containment_pairs.py
