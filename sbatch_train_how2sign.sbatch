#!/bin/bash
#SBATCH --job-name=spamo_how2sign
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=tesi_ztesta
#SBATCH --partition=all_usr_prod
#SBATCH --constraint=gpu_A40_45G|gpu_L40S_45G|gpu_RTX6000_24G|gpu_RTX_A5000_24G

set -euo pipefail

module load anaconda3/2023.09-0-none-none
source activate spamo

cd /homes/ztesta/SpaMo || exit 1

# -----------------------
# W&B OFF + no /tmp issues
# -----------------------
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export WANDB_SILENT=true

export TMPDIR=/work/tesi_ztesta/tmp
export TEMP=/work/tesi_ztesta/tmp
export TMP=/work/tesi_ztesta/tmp
mkdir -p /work/tesi_ztesta/tmp

export TOKENIZERS_PARALLELISM=false

# -----------------------
# Auto-resume from last.ckpt
# -----------------------
LOGROOT="/work/tesi_ztesta/spamo_runs"
TAG="how2sign_flanT5xl_lora_s2"

RESUME_DIR="$(ls -1dt ${LOGROOT}/*_${TAG} 2>/dev/null | head -n 1 || true)"

echo "[INFO] LOGROOT=${LOGROOT}"
echo "[INFO] TAG=${TAG}"
echo "[INFO] RESUME_DIR=${RESUME_DIR:-<none>}"

if [[ -n "${RESUME_DIR}" && -f "${RESUME_DIR}/checkpoints/last.ckpt" ]]; then
  echo "[RESUME] Continuing run: ${RESUME_DIR}"
  python main.py \
    --config configs/finetune_how2sign.yaml \
    --train true \
    --resume "${RESUME_DIR}" \
    --ckpt last.ckpt \
    --no_test true
else
  echo "[START] No checkpoint found, starting new run"
  python main.py \
    --config configs/finetune_how2sign.yaml \
    --train true \
    --logdir "${LOGROOT}" \
    --name "${TAG}" \
    --no_test true
fi
