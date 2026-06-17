#!/bin/bash
#SBATCH --job-name=ar_feat3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_feat3_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
python -u -m src.analysis.ar_intensity.compute_concept_thresholds
python -u -m src.analysis.ar_intensity.build_sae_features
echo PIPELINE DONE
