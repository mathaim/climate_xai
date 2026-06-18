#!/bin/bash
#SBATCH --job-name=ar_clim
#SBATCH --partition=standard
#SBATCH --time=0:20:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_clim_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.climate_concept_map
echo ALL DONE
