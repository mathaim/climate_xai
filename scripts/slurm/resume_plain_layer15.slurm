#!/bin/bash
#SBATCH --job-name=psae_L15r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L15_resume_%j.out
#SBATCH --error=logs/psae_L15_resume_%j.err

# Resume PlainSAE L15 from checkpoint_epoch008.pt (~1 epoch remaining).
# Reads activations + checkpoint from /project; writes new ckpts to /project.

mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"
conda activate py312
cd ~/climate_xai

OUT=/project/AikyamLab/madelyn/GraphCast/train/PlainSAE/Layer15

python -u -m src.training.train_plain_sae \
    --layer 15 \
    --data_dir /project/AikyamLab/madelyn/GraphCast/activations/Layer15 \
    --output_dir "$OUT" \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2 \
    --resume "$OUT/checkpoint_epoch008.pt"

echo "PlainSAE L15 resume done. Models in $OUT"
