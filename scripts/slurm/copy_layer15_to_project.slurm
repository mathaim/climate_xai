#!/bin/bash
#SBATCH --job-name=cp_L15
#SBATCH --partition=standard
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cp_L15_%j.out
#SBATCH --error=logs/cp_L15_%j.err

# Copy GraphCast Layer 15 activations from /standard/AikyamLab (paid, being
# decommissioned) to the new free /project/AikyamLab space.
#
# Single-stream rsync was ~40MB/s (~32h ETA, over the old 24h walltime).
# This runs 8 concurrent rsync streams (one file each) to better saturate
# Ceph, with a 48h walltime as safety margin.
#
# Resumable + verifiable: rsync -a skips files already present & identical in
# the dest, so re-running only copies what is missing. Do NOT delete the
# /standard copy until the file counts at the end of the log match (a
# separate, deliberate step).

mkdir -p ~/climate_xai/logs

SRC=/standard/AikyamLab/madelyn/GraphCast/GraphCastData/layer15
DEST=/project/AikyamLab/madelyn/GraphCast/activations/Layer15

mkdir -p "$DEST"

echo "Source: $SRC"
echo "Dest:   $DEST"
echo "Start:  $(date)"

# Parallel per-file copy. find (not a shell glob) avoids "argument list too
# long" on ~57k files; xargs -P 8 runs 8 rsync streams at once.
find "$SRC" -maxdepth 1 -name '*.npy' -printf '%f\n' \
  | xargs -P 8 -I {} rsync -a "$SRC/{}" "$DEST/"
RC=$?
echo "parallel rsync exit code: $RC"
echo "End:    $(date)"

echo "Source .npy count: $(find "$SRC"  -maxdepth 1 -name '*.npy' | wc -l)"
echo "Dest   .npy count: $(find "$DEST" -maxdepth 1 -name '*.npy' | wc -l)"
echo "If the two counts match and rsync exit code is 0, the copy is complete."
