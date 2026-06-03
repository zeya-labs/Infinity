#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${ROOT_DIR}/outputs"
LATEST_LINK="${RUNS_ROOT}/latest_tokenizer_normal"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT_DEFAULT="${NORMAL_DATA_ROOT:-/root/vepfs/Infinity/data/hypersim/processed/hypersim}"
PN_DEFAULT="${NORMAL_PN:-1M}"
TRAIN_PARTITION_DEFAULT="${NORMAL_TRAIN_PARTITION:-train}"
VAL_PARTITION_DEFAULT="${NORMAL_VAL_PARTITION:-val}"
MAX_TRAIN_SAMPLES_DEFAULT="${NORMAL_MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES_DEFAULT="${NORMAL_MAX_VAL_SAMPLES:-0}"
TRAIN_BATCH_DEFAULT="${NORMAL_TOKENIZER_BATCH_SIZE:-16}"
VAL_BATCH_DEFAULT="${NORMAL_TOKENIZER_VAL_BATCH_SIZE:-16}"
NUM_WORKERS_DEFAULT="${NORMAL_TOKENIZER_NUM_WORKERS:-8}"

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  GPUS="${NPROC_PER_NODE}"
else
  GPUS="$(${PYTHON_BIN} - <<'PY'
import torch
count = torch.cuda.device_count()
print(max(1, count))
PY
)"
fi

RUN_ARGS=("$@")
USE_MANAGED_RUN_DIR=1
for arg in "$@"; do
  case "${arg}" in
    --output-dir=*|--help|-h)
      USE_MANAGED_RUN_DIR=0
      ;;
  esac
done

if [[ "${USE_MANAGED_RUN_DIR}" == "1" ]]; then
  RUN_DATE="$(date '+%Y-%m-%d')"
  RUN_TIME="$(date '+%H-%M-%S')"
  RUN_DIR="${RUNS_ROOT}/${RUN_DATE}/${RUN_TIME}"
  mkdir -p "${RUNS_ROOT}/${RUN_DATE}"
  ln -sfn "${RUN_DIR}" "${LATEST_LINK}"
  RUN_ARGS=(
    --output-dir "${RUN_DIR}"
    --data-root "${DATA_ROOT_DEFAULT}"
    --pn "${PN_DEFAULT}"
    --train-partition "${TRAIN_PARTITION_DEFAULT}"
    --val-partition "${VAL_PARTITION_DEFAULT}"
    --max-train-samples "${MAX_TRAIN_SAMPLES_DEFAULT}"
    --max-val-samples "${MAX_VAL_SAMPLES_DEFAULT}"
    --batch-size "${TRAIN_BATCH_DEFAULT}"
    --val-batch-size "${VAL_BATCH_DEFAULT}"
    --num-workers "${NUM_WORKERS_DEFAULT}"
    "${RUN_ARGS[@]}"
  )
fi

exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${GPUS}" \
  "${ROOT_DIR}/tools/train_normal_tokenizer.py" \
  "${RUN_ARGS[@]}"
