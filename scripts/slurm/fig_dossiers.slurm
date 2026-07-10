#!/bin/bash
#SBATCH --job-name=dossier
#SBATCH --partition=standard
#SBATCH --time=00:50:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/dossier_%j.out
cd /home/euh7ys/climate_xai
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/patching/fig_dossiers.py
