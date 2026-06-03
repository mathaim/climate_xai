#!/bin/bash
#SBATCH --job-name=msae_L0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/msae_L0_%j.out
#SBATCH --error=logs/msae_L0_%j.err

# Train MatryoshkaSAE (BatchTopK) on GraphCast layer 0 activations.
# Reads activations from scratch (fast); writes checkpoints directly to
# standard (durable — survives the 90-day scratch purge). Checkpoint writes
# are small + infrequent, so standard's slow I/O does not affect training speed.

mkdir -p ~/climate_xai/logs

eval "$(conda shell.bash hook)"
conda activate py312

cd ~/climate_xai

OUTPUT_DIR=/standard/AikyamLab/madelyn/GraphCast/MatryoshkaSAE/checkpoints_layer0
mkdir -p "$OUTPUT_DIR"

python -u -m src.training.train_matryoshka_sae \
    --layer 0 \
    --d_model 512 \
    --n_latents 4096 \
    --group_sizes 256 512 1024 2048 4096 \
    --target_l0 32 \
    --n_steps 300000 \
    --batch_size 4096 \
    --lr 3e-2 \
    --data_dir /scratch/euh7ys/climate_xai/activations/layer00 \
    --output_dir "$OUTPUT_DIR" \
    --log_every 500 \
    --save_every 10000 \
    --metrics_every 5000

echo "MatryoshkaSAE L0 training complete. Checkpoints in $OUTPUT_DIR"
