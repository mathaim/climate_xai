#!/bin/bash
#SBATCH --job-name=pool_cmp
#SBATCH --partition=standard
#SBATCH --time=0:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/pool_cmp_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.pool_compare
echo ALL DONE
