#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${ROOT_DIR}/outputs"
LATEST_LINK="${RUNS_ROOT}/latest_tokenizer_normal"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

TRAIN_CACHE_DEFAULT="${NORMAL_TRAIN_CACHE:-/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/train.pt}"
VAL_CACHE_DEFAULT="${NORMAL_VAL_CACHE:-/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/val.pt}"
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
    --train-cache "${TRAIN_CACHE_DEFAULT}"
    --val-cache "${VAL_CACHE_DEFAULT}"
    --batch-size "${TRAIN_BATCH_DEFAULT}"
    --val-batch-size "${VAL_BATCH_DEFAULT}"
    --num-workers "${NUM_WORKERS_DEFAULT}"
    "${RUN_ARGS[@]}"
  )
fi

exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${GPUS}" \
  "${ROOT_DIR}/tools/train_normal_tokenizer.py" \
  "${RUN_ARGS[@]}"
