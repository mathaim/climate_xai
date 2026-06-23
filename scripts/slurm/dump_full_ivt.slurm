#!/bin/bash
#SBATCH --job-name=full_ivt
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/full_ivt_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.dump_full_ivt
echo ALL DONE
