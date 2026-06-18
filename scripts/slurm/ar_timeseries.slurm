#!/bin/bash
#SBATCH --job-name=ar_ts
#SBATCH --partition=standard
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_ts_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.timeseries_concept_ivt
echo ALL DONE
