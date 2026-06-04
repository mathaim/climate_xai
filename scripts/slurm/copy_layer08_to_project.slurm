#!/bin/bash
#SBATCH --job-name=cp_L8
#SBATCH --partition=standard
#SBATCH --time=48:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cp_L8_%j.out
#SBATCH --error=logs/cp_L8_%j.err

# Copy GraphCast Layer 8 TRAIN activations from /scratch (90% full, 90-day
# purge) to the new free /project/AikyamLab space. This both backs up L8
# durably (replacing the AWS plan) and frees ~4.4 TB on scratch for the
# remaining L0 extraction.
#
# Reads from scratch (weka, fast); writes to /project (Ceph). Runs 8 parallel
# rsync streams. Resumable + verifiable: rsync -a skips files already present
# & identical, so re-running only copies what is missing. Do NOT delete the
# scratch copy until the file counts at the end of the log match (a separate,
# deliberate step).

mkdir -p ~/climate_xai/logs

SRC=/scratch/euh7ys/activations_layer08_train
DEST=/project/AikyamLab/madelyn/GraphCast/activations/Layer08

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
echo "If the two counts match, the copy is complete."
