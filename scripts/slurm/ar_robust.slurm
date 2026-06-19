#!/bin/bash
#SBATCH --job-name=ar_robust
#SBATCH --partition=standard
#SBATCH --time=6:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_robust_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=8
python -u -m src.analysis.ar_intensity.timeseries_robustness
echo ALL DONE
