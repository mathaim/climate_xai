#!/bin/bash
#SBATCH --job-name=deadchk
#SBATCH --partition=standard
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/deadchk_%j.out
cd /home/euh7ys/climate_xai
PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u /tmp/check_dead.py
