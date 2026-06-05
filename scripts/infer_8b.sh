#!/usr/bin/env bash

# 使用本地权重执行一次 Infinity 8B 文生图推理。
# 依赖的本地文件：
# - <weights>/infinity_8b_weights
# - <weights>/infinity_vae_d56_f8_14_patchify.pth
# - <weights>/flan-t5-xl
# 主要可调参数：
# - PROMPT：提示词
# - SAVE_FILE：输出图片路径
# - CUDA_VISIBLE_DEVICES：选择使用的 GPU
# - WEIGHTS_DIR / OUTPUT_DIR：权重和输出目录
# 常用方式：
# - CUDA_VISIBLE_DEVICES=0 ./scripts/infer_8b.sh
# - PROMPT="一只在雪地里的红狐狸" SAVE_FILE=./output/fox.png CUDA_VISIBLE_DEVICES=0 ./scripts/infer_8b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_ROOT/weights}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output}"

PROMPT="${PROMPT:-a cinematic portrait of a snow leopard wearing a tailored suit, ultra detailed, photorealistic}"
SAVE_FILE="${SAVE_FILE:-$OUTPUT_DIR/infinity_8b_sample.png}"
PN="${PN:-1M}"
SEED="${SEED:-0}"

export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
export HF_HOME="${HF_HOME:-$WEIGHTS_DIR/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/tools/run_infinity.py" \
  --pn "$PN" \
  --model_type infinity_8b \
  --checkpoint_type torch_shard \
  --model_path "$WEIGHTS_DIR/infinity_8b_weights" \
  --vae_type 14 \
  --vae_path "$WEIGHTS_DIR/infinity_vae_d56_f8_14_patchify.pth" \
  --text_encoder_ckpt "$WEIGHTS_DIR/flan-t5-xl" \
  --text_channels 2048 \
  --cfg 4 \
  --tau 0.5 \
  --use_bit_label 1 \
  --add_lvl_embeding_only_first_block 1 \
  --rope2d_each_sa_layer 1 \
  --rope2d_normalized_by_hw 2 \
  --use_scale_schedule_embedding 0 \
  --apply_spatial_patchify 1 \
  --sampling_per_bits 1 \
  --use_flex_attn 0 \
  --bf16 1 \
  --seed "$SEED" \
  --prompt "$PROMPT" \
  --save_file "$SAVE_FILE"
