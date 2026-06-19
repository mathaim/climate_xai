#!/bin/bash
#SBATCH --job-name=ar_pres
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=3:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_pres_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.compute_nonar_pertimestep
python -u -m src.analysis.ar_intensity.presence_concentration
echo ALL DONE
