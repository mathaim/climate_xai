#!/bin/bash
#SBATCH --job-name=cp_L15
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/cp_L15_%j.out
#SBATCH --error=logs/cp_L15_%j.err

# Copy GraphCast Layer 15 activations from /standard/AikyamLab (paid, being
# decommissioned) to the new free /project/AikyamLab space.
#
# rsync is resumable + verifiable: if this job is killed or re-run, it only
# transfers what is missing/changed. Do NOT delete the /standard copy until
# the file counts at the end of the log match (a separate, deliberate step).

mkdir -p ~/climate_xai/logs

SRC=/standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer15
DEST=/project/AikyamLab/madelyn/GraphCast/activations/Layer15

mkdir -p "$DEST"

echo "Source: $SRC"
echo "Dest:   $DEST"
echo "Start:  $(date)"

# -a preserve everything; --info progress/stats for a readable log.
# Trailing slash on SRC copies the *contents* of layer15 into Layer15.
rsync -a --info=progress2,stats2 "$SRC/" "$DEST/"
RC=$?
echo "rsync exit code: $RC"
echo "End:    $(date)"

echo "Source .npy count: $(find "$SRC"  -maxdepth 1 -name '*.npy' | wc -l)"
echo "Dest   .npy count: $(find "$DEST" -maxdepth 1 -name '*.npy' | wc -l)"
echo "If the two counts match and rsync exit code is 0, the copy is complete."
