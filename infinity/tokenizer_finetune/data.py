from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class HypersimNormalCacheDataset(Dataset):
    """Memory-mapped normal-map cache exported by NormalART."""

    def __init__(self, cache_path: str, repeat: int = 1, mmap: bool = True) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Tokenizer cache not found: {self.cache_path}")
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")

        payload = torch.load(
            self.cache_path,
            map_location="cpu",
            weights_only=False,
            mmap=mmap,
        )
        self.targets = payload["targets"].contiguous()
        self.masks = payload["masks"].contiguous()
        self.metadata = payload.get("metadata", [])
        self.partition = payload.get("partition", "train")
        self.repeat = repeat
        self.num_samples = int(self.targets.shape[0])

        if self.masks.shape[0] != self.num_samples:
            raise ValueError("targets and masks do not share the same first dimension")

    def __len__(self) -> int:
        return self.num_samples * self.repeat

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = index % self.num_samples
        metadata = self.metadata[sample_index] if self.metadata else {}
        return {
            "target": self.targets[sample_index],
            "mask": self.masks[sample_index],
            "metadata": metadata,
        }


def collate_normal_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "mask": torch.stack([sample["mask"] for sample in samples], dim=0),
        "metadata": [sample.get("metadata", {}) for sample in samples],
    }
