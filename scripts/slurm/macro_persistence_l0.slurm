#!/bin/bash
#SBATCH --job-name=persl0
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/persl0_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000 REQUIRE_GPU=1
SRC=matry_L0 DST=matry_L8  OUT=/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence_L0L8.npz  PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
SRC=matry_L0 DST=matry_L15 OUT=/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence_L0L15.npz PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
