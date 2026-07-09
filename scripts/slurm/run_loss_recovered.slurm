#!/bin/bash
#SBATCH --job-name=run_loss_recovered
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/euh7ys/climate_xai/logs/run_loss_recovered_%j.out
module load cuda/12.8.0
source /home/euh7ys/ondemand/Research/GraphCast/graphcast-interpretability/graphcast_sae_env/bin/activate
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
cd /home/euh7ys/climate_xai
export GRAPHCAST_CHECKPOINT="/scratch/euh7ys/climate_xai/graphcast_checkpoints/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
PYTHONPATH=. python -u -m src.patching.run_loss_recovered
echo ALL DONE
