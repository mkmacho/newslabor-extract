#!/bin/bash
# SLURM job script for the address-resolution stage.
#
# Submitted as a job array, one task per newspaper:
#
#   export GEOAPIFY_API_KEY=...
#   sbatch --array=0-12%1 batch.sh
#
# The %1 serialises the array. Without it, thirteen tasks issue requests
# concurrently and the per-process rate limit no longer bounds the total rate
# seen by the API. Serialise, or drop --rate_limit proportionally.
#
# Override paths with DATA_DIR / OUT_DIR; both default to the repo layout.

#SBATCH --job-name=resolve
## Slurm opens this file before the job body runs, so it cannot depend on a
## directory the script itself creates. Written beside the submission directory;
## pass --output=logs/... on the sbatch command line if you have made logs/ first.
#SBATCH --output=resolve-%A_%a.log        ## %A job id, %a array index
#SBATCH --partition=high                  ## high/low/gpu; default is low
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20                ## threads used by --nworkers below
#SBATCH --mem-per-cpu=2G                  ## response payloads are held per batch
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END,FAIL              ## set --mail-user at submit time, e.g.
                                          ## sbatch --mail-user=you@example.edu

set -euo pipefail

PAPERS=(ASA ATC ATL BaS BoG ChT HaC LAS LAT NJG NYr NYT WaP)
PAPER="${PAPERS[${SLURM_ARRAY_TASK_ID:-9}]}"     # default index 9 = NJG

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$REPO/output}"
OUT_DIR="${OUT_DIR:-$REPO/output}"
NWORKERS="${SLURM_CPUS_PER_TASK:-4}"

: "${GEOAPIFY_API_KEY:?set GEOAPIFY_API_KEY before submitting}"

mkdir -p "$OUT_DIR"
module load python 2>/dev/null || true

echo "Resolving ${PAPER} with ${NWORKERS} threads on $(hostname) at $(date)"

srun python "$REPO/scripts/resolve.py" \
  --filepath="$DATA_DIR/${PAPER}-extract-all.gzip" \
  --aux_dir="$REPO/auxiliary_files" \
  --output_dir="$OUT_DIR" \
  --nworkers="$NWORKERS" \
  --multithreading=1 \
  --rate_limit=10

echo "Finished ${PAPER} at $(date)"
