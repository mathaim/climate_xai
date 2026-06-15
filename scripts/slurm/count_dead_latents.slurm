#!/bin/bash
#SBATCH --job-name=dead_sae
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/dead_sae_%j.out
#SBATCH --error=logs/dead_sae_%j.err
mkdir -p ~/climate_xai/logs
eval "$(conda shell.bash hook)"; conda activate py312
cd ~/climate_xai
P=/project/AikyamLab/madelyn/GraphCast
A0=/scratch/euh7ys/climate_xai/activations/layer00
A8=$P/activations/Layer08
A15=$P/activations/Layer15
run(){ echo "================ $1 ================"; python -u -m src.analysis.count_dead_latents "${@:2}"; }
run "Plain L0"       --arch plain      --layer 0  --checkpoint $P/train/PlainSAE/Layer00/final_model.pt --data_dir $A0
run "Plain L8"       --arch plain      --layer 8  --checkpoint /scratch/euh7ys/climate_xai/checkpoints/plain_layer8/final_model.pt --data_dir $A8
run "Plain L15"      --arch plain      --layer 15 --checkpoint $P/train/PlainSAE/Layer15/checkpoint_epoch008.pt --data_dir $A15
run "Matryoshka L0"  --arch matryoshka --layer 0  --checkpoint $P/train/MatryoshkaSAE/Layer00/final_model.pt --data_dir $A0  --topk_mode per_sample
run "Matryoshka L8"  --arch matryoshka --layer 8  --checkpoint $P/train/MatryoshkaSAE/Layer08/final_model.pt --data_dir $A8  --topk_mode per_sample
run "Matryoshka L15" --arch matryoshka --layer 15 --checkpoint $P/train/MatryoshkaSAE/Layer15/final_model.pt --data_dir $A15 --topk_mode per_sample
echo "ALL DONE"
