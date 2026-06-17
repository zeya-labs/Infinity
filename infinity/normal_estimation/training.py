from __future__ import annotations

from typing import Any

import torch


def require_positive_steps_per_epoch(steps_per_epoch: int, *, context: str) -> int:
    if steps_per_epoch <= 0:
        raise RuntimeError(
            f"{context} has no training batches. "
            "Check dataset size, batch size, world size, drop_last, and target-size grouping."
        )
    return steps_per_epoch


def normalize_normal_tensor(normals: torch.Tensor) -> torch.Tensor:
    return normals.float() / torch.linalg.norm(normals.float(), dim=1, keepdim=True).clamp_min(1e-6)


def convert_ar_target_to_eval_convention(target: torch.Tensor, dataset_name: str) -> torch.Tensor:
    dataset_name = dataset_name.lower()
    if dataset_name == "nyuv2":
        return normalize_normal_tensor(torch.stack((-target[:, 0], target[:, 2], -target[:, 1]), dim=1))
    return normalize_normal_tensor(target)


def convert_ar_batch_target_to_eval_convention(target: torch.Tensor, metadata: list[dict[str, Any]]) -> torch.Tensor:
    dataset_names = {str(item.get("dataset", "hypersim")).lower() for item in metadata}
    if len(dataset_names) != 1:
        raise ValueError(f"AR eval batches must contain one dataset convention, got {sorted(dataset_names)}")
    return convert_ar_target_to_eval_convention(target, next(iter(dataset_names)))
