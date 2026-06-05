from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HYPERSIM_ROOT = str(REPO_ROOT / "data" / "hypersim" / "processed" / "hypersim")
DEFAULT_VKITTI2_ROOT = str(REPO_ROOT / "data" / "VKITTI2")
DEFAULT_NORMAL_TRAIN_DATASETS = "hypersim,vkitti2"
DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS = "hypersim:3,vkitti2:1"

DEFAULT_NORMAL_TOKENIZER_CKPT = os.environ.get(
    "INFINITY_NORMAL_TOKENIZER_CKPT",
    str(REPO_ROOT / "outputs" / "normal_tokenizer" / "2026-06-03" / "00-39-35" / "checkpoints" / "best_angle_3.5732.pth"),
)
LEGACY_NORMAL_TOKENIZER_CKPT = os.environ.get(
    "INFINITY_LEGACY_NORMAL_TOKENIZER_CKPT",
    str(REPO_ROOT / "outputs" / "normal_tokenizer" / "2026-05-31" / "15-08-43" / "checkpoints" / "best_angle_6.7867.pth"),
)
DEFAULT_NORMAL_ESTIMATION_CKPT = os.environ.get(
    "INFINITY_NORMAL_ESTIMATION_CKPT",
    str(REPO_ROOT / "outputs" / "normal_estimation" / "2026-06-01" / "09-27-05" / "checkpoints" / "best_angle_18.5532.pth"),
)
