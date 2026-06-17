from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERSIM_ROOT = str(REPO_ROOT / "data" / "hypersim" / "processed" / "hypersim")
DEFAULT_VKITTI2_ROOT = str(REPO_ROOT / "data" / "VKITTI2" / "processed" / "normals_lotus_svd")
DEFAULT_NORMAL_TRAIN_DATASETS = "hypersim,vkitti2"
DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS = "hypersim:9,vkitti2:1"
DEFAULT_VKITTI2_MAX_INVALID_RATIO = 0.25

DEFAULT_NORMAL_TOKENIZER_CKPT = os.environ.get(
    "INFINITY_NORMAL_TOKENIZER_CKPT",
    str(REPO_ROOT / "outputs" / "normal_tokenizer" / "2026-06-08" / "07-06-39" / "checkpoints" / "best_angle_2.9407.pth"),
)
DEFAULT_NORMAL_ESTIMATION_CKPT = os.environ.get(
    "INFINITY_NORMAL_ESTIMATION_CKPT",
    str(REPO_ROOT / "outputs" / "normal_estimation" / "2026-06-07" / "06-24-56" / "checkpoints" / "best_angle_13.2903.pth"),
)
