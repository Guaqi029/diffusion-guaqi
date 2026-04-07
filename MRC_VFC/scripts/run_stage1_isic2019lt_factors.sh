#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64,garbage_collection_threshold:0.8}"

DATA_ROOT="${DATA_ROOT:-/data/DataLACP/guyiqin/ISIC2019LT/ISIC_2019_Training_Input}"
LT_SPLIT_DIR="${LT_SPLIT_DIR:-/data/DataLACP/guyiqin/CODE/Difussion/MRC_VFC/split/ISIC2019LT/shared_eval_seed42}"
LT_SPLIT_ROOT="$(dirname "${LT_SPLIT_DIR}")"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-./runs_stage1_factor_sweep}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-100}"
EVAL_EVERY="${EVAL_EVERY:-5}"
TRAIN_LOG_EVERY="${TRAIN_LOG_EVERY:-50}"
CONSOLE_LOG_EVERY="${CONSOLE_LOG_EVERY:-50}"

VAVAE_INIT="${VAVAE_INIT:-./checkpoints/pretrained/yao_vavae/vavae-imagenet256-f16d32-dinov2.pt}"

check_split_files() {
  local factor="$1"
  local train_csv="${LT_SPLIT_DIR}/training_if${factor}.csv"
  local val_csv="${LT_SPLIT_DIR}/validation.csv"
  local test_csv="${LT_SPLIT_DIR}/testing.csv"

  [[ -f "${train_csv}" ]] || { echo "Missing split file: ${train_csv}" >&2; exit 1; }
  [[ -f "${val_csv}" ]] || { echo "Missing split file: ${val_csv}" >&2; exit 1; }
  [[ -f "${test_csv}" ]] || { echo "Missing split file: ${test_csv}" >&2; exit 1; }
}

run_factor() {
  local factor="$1"
  local timestamp
  local mark
  local run_name
  local run_dir
  local run_log

  check_split_files "${factor}"

  timestamp="$(date +"%Y%m%d_%H%M%S")"
  mark="isic2019lt_if${factor}"
  run_name="run_s1_${mark}_${timestamp}"
  run_dir="${RUN_ROOT}/ISIC2019LT/${mark}"
  run_log="${run_dir}/train_${timestamp}.log"
  mkdir -p "${run_dir}"

  {
    echo "[${timestamp}] Start factor=${factor}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "DATA_ROOT=${DATA_ROOT}"
    echo "LT_SPLIT_DIR=${LT_SPLIT_DIR}"
    echo "TRAIN_CSV=${LT_SPLIT_DIR}/training_if${factor}.csv"
    echo "VAL_CSV=${LT_SPLIT_DIR}/validation.csv"
    echo "TEST_CSV=${LT_SPLIT_DIR}/testing.csv"
    echo "SEED=${SEED}"
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
    echo "NUM_WORKERS=${NUM_WORKERS}"
    echo "EPOCHS=${EPOCHS}"
    echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    echo "RUN_NAME=${run_name}"
    echo "LOG_FILE=${run_log}"
    echo ""
  } | tee -a "${run_log}"

  python stage1.py --debug \
    --run_name "${run_name}" \
    --student_run_name "${run_name}" \
    --reload False \
    --dataset ISIC2019LT \
    --data_path "${DATA_ROOT}" \
    --checkpoints "${CHECKPOINT_ROOT}" \
    --imbalance_factor "${factor}" \
    --seed "${SEED}" \
    --lt_split_root "${LT_SPLIT_ROOT}" \
    --gpus 1 \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --workers "${NUM_WORKERS}" \
    --student_source vavae \
    --vavae_student_init_path "${VAVAE_INIT}" \
    --vavae_student_latent_dim 32 \
    --vavae_student_enable_decoder False \
    --vavae_student_input_size 224 \
    --vavae_student_resize_input True \
    --kd_enable True --kd_only True --kd_freeze_teacher True \
    --kd_teacher_source lite \
    --kd_lite_teacher_use_weak_aug True \
    --kd_lite_teacher_use_ema True --kd_lite_teacher_ema_decay 0.99 \
    --kd_feat_project False \
    --kd_logit_weight 0.5 \
    --kd_feat_weight 0.5 --kd_feat_start_epoch 0 \
    --kd_struct_type cka --kd_struct_weight 1.0 --kd_struct_start_epoch 20 \
    --lite_vae_recon_weight 0.0 --lite_vae_kl_weight 0.0 \
    --lite_student_ce_weight 1.0 \
    --show_teacher_metrics False \
    --use_class_weight True \
    --stage1_cls_loss_type ce \
    --stage1_drw_enable False \
    --epochs "${EPOCHS}" \
    --eval_every_epochs "${EVAL_EVERY}" \
    --train_log_every_iters "${TRAIN_LOG_EVERY}" \
    --console_log_every_iters "${CONSOLE_LOG_EVERY}" \
    --log_file "${run_log}" 2>&1 | tee -a "${run_log}"

  echo "[$(date +"%Y%m%d_%H%M%S")] Finished factor=${factor} run=${run_name}" | tee -a "${run_log}"
}

run_factor 100
run_factor 200
run_factor 500
