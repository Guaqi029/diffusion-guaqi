#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.8}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-12356}"

DATA_ROOT="${DATA_ROOT:-/mnt/c/Users/guyiq/Desktop/ISIC_Archive}"
SPLIT_DIR="${SPLIT_DIR:-/mnt/c/Users/guyiq/Desktop/kuosan-mrc/MRC_VFC/split/ISIC_Archive}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"
RUN_ROOT="${RUN_ROOT:-./runs_stage1_isic_archive}"

SEED="${SEED:-42}"
GPUS="${GPUS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-5}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
CONSOLE_LOG_EVERY="${CONSOLE_LOG_EVERY:-50}"

TRAIN_CSV="${SPLIT_DIR}/training.csv"
VAL_CSV="${SPLIT_DIR}/validation.csv"
TEST_CSV="${SPLIT_DIR}/testing.csv"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data dir: ${DATA_ROOT}" >&2; exit 1; }
[[ -f "${TRAIN_CSV}" ]] || { echo "Missing split file: ${TRAIN_CSV}" >&2; exit 1; }
[[ -f "${VAL_CSV}" ]] || { echo "Missing split file: ${VAL_CSV}" >&2; exit 1; }
[[ -f "${TEST_CSV}" ]] || { echo "Missing split file: ${TEST_CSV}" >&2; exit 1; }

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RUN_NAME="${RUN_NAME:-run_s1_isic_archive_${TIMESTAMP}}"
RUN_DIR="${RUN_ROOT}/ISIC_Archive/base"
RUN_LOG="${RUN_DIR}/train_${TIMESTAMP}.log"
mkdir -p "${RUN_DIR}"

{
  echo "[${TIMESTAMP}] Start ISIC_Archive Stage1"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "GPUS=${GPUS}"
  echo "MASTER_ADDR=${MASTER_ADDR}"
  echo "MASTER_PORT=${MASTER_PORT}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "TRAIN_CSV=${TRAIN_CSV}"
  echo "VAL_CSV=${VAL_CSV}"
  echo "TEST_CSV=${TEST_CSV}"
  echo "SEED=${SEED}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
  echo "NUM_WORKERS=${NUM_WORKERS}"
  echo "EPOCHS=${EPOCHS}"
  echo "CHECKPOINT_ROOT=${CHECKPOINT_ROOT}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
  echo "LOG_FILE=${RUN_LOG}"
  echo ""
} | tee -a "${RUN_LOG}"

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
  --gpus "${GPUS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --eval_every_epochs "${EVAL_EVERY}" \
  --train_log_every_iters "${TRAIN_LOG_EVERY}" \
  --console_log_every_iters "${CONSOLE_LOG_EVERY}" \
  --log_file "${RUN_LOG}" 2>&1 | tee -a "${RUN_LOG}"

echo "[$(date +"%Y%m%d_%H%M%S")] Finished run=${RUN_NAME}" | tee -a "${RUN_LOG}"
