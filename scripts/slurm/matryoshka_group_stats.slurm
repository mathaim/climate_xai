#!/bin/bash
#SBATCH --job-name=msae_groups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/msae_groups_%j.out
#SBATCH --error=logs/msae_groups_%j.err
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate py312
cd ~/climate_xai
P=/project/AikyamLab/madelyn/GraphCast
N=5698
python -u -m src.analysis.matryoshka_group_stats --layer 0  --checkpoint $P/train/MatryoshkaSAE/Layer00/final_model.pt --data_dir /scratch/euh7ys/climate_xai/activations/layer00 --n_files $N
python -u -m src.analysis.matryoshka_group_stats --layer 8  --checkpoint $P/train/MatryoshkaSAE/Layer08/final_model.pt --data_dir $P/activations/Layer08 --n_files $N
python -u -m src.analysis.matryoshka_group_stats --layer 15 --checkpoint $P/train/MatryoshkaSAE/Layer15/final_model.pt --data_dir $P/activations/Layer15 --n_files $N
echo "ALL DONE"
