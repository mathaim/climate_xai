#!/bin/bash
#SBATCH --job-name=ar_fp_cpu
#SBATCH --partition=standard
#SBATCH --time=0:20:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_fp_cpu_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.mean_footprint
python -u -m src.analysis.ar_intensity.footprint_figure
echo ALL DONE
