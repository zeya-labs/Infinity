#!/usr/bin/env bash

# 下载 Infinity 8B 推理所需的最小本地资源。
# 下载内容：
# - FoundationVision/Infinity：8B 模型分片 + 14-patchify VAE
# - google/flan-t5-xl：文本编码器权重和分词器文件
# 默认存储位置：
# - WEIGHTS_DIR，默认：<repo>/weights
# - HF 缓存，默认：<repo>/weights/.cache/huggingface
# 常用方式：
# - ./scripts/setup_8b_infer.sh
# - WEIGHTS_DIR=/path/to/weights ./scripts/setup_8b_infer.sh
# - env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY ./scripts/setup_8b_infer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$REPO_ROOT/weights}"
HF_HOME="${HF_HOME:-$WEIGHTS_DIR/.cache/huggingface}"

mkdir -p "$WEIGHTS_DIR" "$HF_HOME"

export WEIGHTS_DIR
export HF_HOME
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"

"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

weights_dir = os.environ["WEIGHTS_DIR"]

snapshot_download(
    repo_id="FoundationVision/Infinity",
    allow_patterns=[
        "infinity_8b_weights/*",
        "infinity_vae_d56_f8_14_patchify.pth",
    ],
    local_dir=weights_dir,
)

snapshot_download(
    repo_id="google/flan-t5-xl",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    local_dir=os.path.join(weights_dir, "flan-t5-xl"),
)
PY
