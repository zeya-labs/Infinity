#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH_DEFAULT="${NORMAL_MODEL_PATH:-${ROOT_DIR}/outputs/latest_normal_estimation/checkpoints/last.pth}"
OUTPUT_DIR_DEFAULT="${NORMAL_OUTPUT_DIR:-${ROOT_DIR}/outputs/normal_predictions}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/tools/run_normal_estimation.py" \
  --model-path "${MODEL_PATH_DEFAULT}" \
  --output-dir "${OUTPUT_DIR_DEFAULT}" \
  "$@"
