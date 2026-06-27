#!/bin/bash
#SBATCH --job-name=clim_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=1:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/clim_map_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.climate_concept_map
echo ALL DONE
