#!/bin/bash
#SBATCH --job-name=ar_analyze
#SBATCH --partition=standard
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/euh7ys/climate_xai/logs/ar_analyze_%j.out
cd /home/euh7ys/climate_xai
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=8
echo "===== B: decodability ====="
python -u -m src.analysis.ar_intensity.analyze_decode
echo "===== A: concept tracking ====="
python -u -m src.analysis.ar_intensity.analyze_concepts
echo "ALL DONE"
