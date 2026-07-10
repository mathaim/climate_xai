#!/bin/bash
#SBATCH --job-name=d1592
#SBATCH --partition=standard
#SBATCH --time=00:50:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/d1592_%j.out
cd /home/euh7ys/climate_xai
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/patching/fig_dossier_1592.py
