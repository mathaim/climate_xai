#!/bin/bash
#SBATCH --job-name=persp
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:a6000:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/euh7ys/climate_xai/logs/persp_%j.out
cd /home/euh7ys/climate_xai
export NMAX=8000 REQUIRE_GPU=1
SRC=plain_L8 DST=plain_L15 OUT=/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence_plain_L8L15.npz PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
SRC=plain_L0 DST=plain_L8  OUT=/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence_plain_L0L8.npz  PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
SRC=plain_L0 DST=plain_L15 OUT=/scratch/euh7ys/climate_xai/concept_ivt/macro_persistence_plain_L0L15.npz PYTHONPATH=. /home/euh7ys/miniconda3/bin/python -u src/analysis/ar_intensity/macro_persistence.py
