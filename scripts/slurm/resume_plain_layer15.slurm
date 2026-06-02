#!/bin/bash
#SBATCH --job-name=psae_L15r
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/psae_L15_resume_%j.out
#SBATCH --error=logs/psae_L15_resume_%j.err

# Resume PlainSAE L15 from checkpoint_epoch008.pt (epoch 9 done, need epoch 10).
# Only 1 epoch remaining — 4h is plenty.

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

python -u -m src.training.train_plain_sae \
    --layer 15 \
    --data_dir /scratch/euh7ys/climate_xai/activations/layer15 \
    --output_dir /scratch/euh7ys/climate_xai/checkpoints/plain_layer15 \
    --d_in 512 \
    --n_latents 4096 \
    --k_active 32 \
    --batch_size 8192 \
    --lr 2e-4 \
    --epochs 10 \
    --save_every 2 \
    --resume /scratch/euh7ys/climate_xai/checkpoints/plain_layer15/checkpoint_epoch008.pt

# Copy final model to standard storage
STANDARD_DIR=/standard/AikyamLab/madelyn/GraphCast/PlainSAE/checkpoints_layer15
mkdir -p "$STANDARD_DIR"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer15/final_model.pt "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer15/config.json "$STANDARD_DIR/"
cp /scratch/euh7ys/climate_xai/checkpoints/plain_layer15/training_log.jsonl "$STANDARD_DIR/"
echo "Final model copied to $STANDARD_DIR"
