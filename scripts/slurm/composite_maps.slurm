#!/bin/bash
#SBATCH --job-name=comp_map
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=0:50:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/comp_map_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.composite_intensity_maps
echo ALL DONE
