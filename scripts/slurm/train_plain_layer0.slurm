#!/bin/bash
#SBATCH --job-name=psae_L0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L0_%j.out
#SBATCH --error=logs/psae_L0_%j.err
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"
conda activate py312
cd ~/climate_xai
OUT=/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer00
mkdir -p "$OUT"
python -u -m src.training.train_plain_sae \
    --layer 0 \
    --data_dir /scratch/euh7ys/climate_xai/activations/layer00 \
    --output_dir "$OUT" \
    --d_in 512 --n_latents 4096 --k_active 32 --batch_size 8192 \
    --lr 2e-4 --epochs 10 --save_every 2
echo "PlainSAE L0 done. Models in $OUT"
