#!/bin/bash

set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.8}"

DATA_ROOT="${DATA_ROOT:-/mnt/c/Users/guyiq/Desktop/ISIC_Archive}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/split/ISIC_Archive}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
LOG_ROOT="${LOG_ROOT:-./log/stage1_eval}"

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_COUNT="${GPU_COUNT:-1}"
PRINT_PER_CLASS="${PRINT_PER_CLASS:-False}"

RUN_NAME="${RUN_NAME:-}"

resolve_run_name() {
  if [[ -n "${RUN_NAME}" ]]; then
    echo "${RUN_NAME}"
    return 0
  fi

  local pattern="${CHECKPOINT_ROOT}/run_s1_isic_archive_*"
  local matches=( ${pattern} )
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo ""
    return 0
  fi
  ls -dt ${pattern} | head -n 1 | xargs basename
}

TRAIN_CSV="${SPLIT_DIR}/training.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/testing.csv"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

RUN_NAME="$(resolve_run_name)"
[[ -n "${RUN_NAME}" ]] || { echo "Could not resolve RUN_NAME for ISIC_Archive. Set RUN_NAME explicitly." >&2; exit 1; }
[[ -d "${CHECKPOINT_ROOT}/${RUN_NAME}" ]] || { echo "Missing checkpoint dir: ${CHECKPOINT_ROOT}/${RUN_NAME}" >&2; exit 1; }

mkdir -p "${LOG_ROOT}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_ROOT}/${RUN_NAME}_archive_${TIMESTAMP}.log"

{
  echo "[${TIMESTAMP}] Evaluate Stage1 ISIC_Archive"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "SPLIT_DIR=${SPLIT_DIR}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "CHECKPOINT_DIR=${CHECKPOINT_ROOT}/${RUN_NAME}"
  echo "TRAIN_CSV=${TRAIN_CSV}"
  echo "VAL_CSV=${VAL_CSV}"
  echo "TEST_CSV=${TEST_CSV}"
  echo "SEED=${SEED}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "PRINT_PER_CLASS=${PRINT_PER_CLASS}"
  echo "LOG_FILE=${LOG_FILE}"
  echo ""
} | tee -a "${LOG_FILE}"

python stage1.py --debug \
  --run_name "${RUN_NAME}" \
  --student_run_name "${RUN_NAME}" \
  --dataset ISIC_Archive \
  --data_path "${DATA_ROOT}" \
  --csv_file_train "${TRAIN_CSV}" \
  --csv_file_val "${VAL_CSV}" \
  --csv_file_test "${TEST_CSV}" \
  --checkpoints "${CHECKPOINT_ROOT}" \
  --seed "${SEED}" \
  --gpus "${GPU_COUNT}" \
  --batch_size "${BATCH_SIZE}" \
  --workers "${NUM_WORKERS}" \
  --reload False \
  --lite_eval_enable True \
  --lite_eval_only True \
  --lite_eval_use_classifier True \
  --stage1_log_per_class_metrics "${PRINT_PER_CLASS}" \
  --lite_vae_resume_path litevae_latest.pth \
  --lite_classifier_resume_path lite_classifier_latest.pth \
  --log_file "${LOG_FILE}" 2>&1 | tee -a "${LOG_FILE}"
