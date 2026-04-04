#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${ROOT_DIR}/outputs"
LATEST_LINK="${RUNS_ROOT}/latest_normal_estimation"
PYTHON_BIN="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT_DEFAULT="${NORMAL_DATA_ROOT:-/root/vepfs/NormalART/datasets/processed/hypersim}"
NORMAL_VAE_CKPT_FALLBACK="${ROOT_DIR}/weights/infinity_vae_d56_f8_14_patchify.pth"
NORMAL_VAE_CKPT_DEFAULT="${NORMAL_VAE_CKPT:-${NORMAL_VAE_CKPT_FALLBACK}}"

RGB_VAE_CKPT_FALLBACK="${ROOT_DIR}/weights/infinity_vae_d56_f8_14_patchify.pth"
RGB_VAE_CKPT_DEFAULT="${RGB_VAE_CKPT:-${RGB_VAE_CKPT_FALLBACK}}"
INIT_MODEL_DEFAULT="${INIT_MODEL_PATH:-${ROOT_DIR}/weights/infinity_8b_weights}"
MODEL_NAME_DEFAULT="${NORMAL_MODEL_NAME:-infinity_8b}"
TRAIN_BATCH_DEFAULT="${NORMAL_BATCH_SIZE:-4}"
VAL_BATCH_DEFAULT="${NORMAL_VAL_BATCH_SIZE:-4}"
NUM_WORKERS_DEFAULT="${NORMAL_NUM_WORKERS:-4}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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

if [[ -n "${NORMAL_ZERO:-}" ]]; then
  ZERO_DEFAULT="${NORMAL_ZERO}"
elif [[ "${MODEL_NAME_DEFAULT}" == "infinity_8b" && "${GPUS}" -gt 1 ]]; then
  ZERO_DEFAULT=3
else
  ZERO_DEFAULT=0
fi

ENABLE_HYBRID_SHARD_DEFAULT="${NORMAL_ENABLE_HYBRID_SHARD:-0}"
INNER_SHARD_DEGREE_DEFAULT="${NORMAL_INNER_SHARD_DEGREE:-1}"
FSDP_USE_ORIG_PARAMS_DEFAULT="${NORMAL_FSDP_USE_ORIG_PARAMS:-1}"

RUN_ARGS=("$@")
USE_MANAGED_RUN_DIR=1
for arg in "$@"; do
  case "${arg}" in
    --output-dir=*|--output-dir|--help|-h)
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
    --normal-vae-ckpt "${NORMAL_VAE_CKPT_DEFAULT}"
    --rgb-vae-ckpt "${RGB_VAE_CKPT_DEFAULT}"
    --normal-vae-type 14
    --rgb-vae-type 14
    --normal-apply-spatial-patchify
    --rgb-apply-spatial-patchify
    --model-name "${MODEL_NAME_DEFAULT}"
    --init-model "${INIT_MODEL_DEFAULT}"
    --batch-size "${TRAIN_BATCH_DEFAULT}"
    --val-batch-size "${VAL_BATCH_DEFAULT}"
    --num-workers "${NUM_WORKERS_DEFAULT}"
    --zero "${ZERO_DEFAULT}"
    --inner-shard-degree "${INNER_SHARD_DEGREE_DEFAULT}"
    "${RUN_ARGS[@]}"
  )
  if [[ "${ENABLE_HYBRID_SHARD_DEFAULT}" == "1" ]]; then
    RUN_ARGS+=(--enable-hybrid-shard)
  fi
  if [[ "${FSDP_USE_ORIG_PARAMS_DEFAULT}" != "1" ]]; then
    RUN_ARGS+=(--disable-fsdp-use-orig-params)
  fi
fi

exec "${TORCHRUN_BIN}" --standalone --nproc_per_node="${GPUS}" \
  "${ROOT_DIR}/tools/train_normal_estimation.py" \
  "${RUN_ARGS[@]}"
