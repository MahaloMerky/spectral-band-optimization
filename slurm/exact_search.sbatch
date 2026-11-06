#!/bin/bash
# Usage, from the repository root:
#   mkdir -p logs
#   STATS=data/processed/class_stats.npz K=4 sbatch slurm/exact_search.sbatch
# Then merge:
#   python scripts/merge_exact.py results/exact/k$K --stats $STATS
#
# Set the partition and any module loads for your cluster before the first run.

#SBATCH --job-name=exact-search
#SBATCH --partition=CHANGE_ME
#SBATCH --array=0-31
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

: "${STATS:?Set STATS to the class statistics file}"
: "${K:?Set K to the band count}"

python scripts/exact_search.py --stats "$STATS" --k "$K" --out "results/exact/k$K"
