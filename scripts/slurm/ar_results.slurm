#!/bin/bash
#SBATCH --job-name=ar_results
#SBATCH --partition=standard
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_results_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1
python -u -m src.analysis.ar_intensity.compute_results
echo RESULTS DONE
