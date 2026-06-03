from __future__ import annotations

from typing import Any

import torch


def collate_normal_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "mask": torch.stack([sample["mask"] for sample in samples], dim=0),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }
