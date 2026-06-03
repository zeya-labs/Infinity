from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

from .file_io import read_depth_normal_hypersim


def _nearest_h_div_w_template(height: int, width: int) -> float:
    ratio = float(height) / float(width)
    return float(h_div_w_templates[np.argmin(np.abs(h_div_w_templates - ratio))])


def _resolve_target_size(height: int, width: int, pn: str) -> tuple[int, int, float]:
    template = _nearest_h_div_w_template(height, width)
    target_height, target_width = dynamic_resolution_h_w[template][pn]["pixel"]
    return int(target_height), int(target_width), template


def _resize_image(image: Image.Image, target_hw: tuple[int, int]) -> torch.Tensor:
    target_height, target_width = target_hw
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1)


def _resize_tensor(
    tensor: torch.Tensor,
    target_hw: tuple[int, int],
    *,
    mode: str,
    normalize_normals: bool = False,
) -> torch.Tensor:
    if tensor.shape[-2:] == target_hw:
        return tensor

    kwargs: dict[str, Any] = {"size": target_hw, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    tensor = F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)
    if normalize_normals:
        tensor = tensor / torch.linalg.norm(tensor, dim=0, keepdim=True).clamp_min(1e-6)
    return tensor


def _compute_hypersim_intrinsics(fov_value: float, pixel_height: int, pixel_width: int) -> list[float]:
    # Mirror the conversion used by NormalART's raw Hypersim reader for compatibility.
    fov_y_rad = float(fov_value)
    fov_x_rad = 2 * np.arctan(pixel_width * np.tan(fov_y_rad / 2.0) / pixel_height)
    fx = (pixel_width / 2.0) / np.tan(fov_y_rad / 2.0)
    fy = (pixel_height / 2.0) / np.tan(fov_x_rad / 2.0)
    cx = pixel_width / 2.0
    cy = pixel_height / 2.0
    return [float(fx), float(cx), float(fy), float(cy)]


class HypersimNormalDataset(Dataset):
    """Read raw Hypersim jpg+hdf5 samples for RGB-to-normal estimation."""

    def __init__(
        self,
        root: str,
        partition: str = "train",
        pn: str = "0.06M",
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"partition must be one of train/val/test, got {partition}")

        self.root = Path(root)
        csv_path = self.root / f"final_{partition}_split.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Hypersim split CSV not found: {csv_path}")
        if pn not in {"0.06M", "0.25M", "1M"}:
            raise ValueError(f"Unsupported pn: {pn}")

        self.partition = partition
        self.pn = pn
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self.records = list(reader)
        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.root / record["images"]
        depth_path = self.root / record["depth"]
        normal_path = self.root / record["normal"]

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            original_width, original_height = image.size
            image_tensor = _resize_image(
                image,
                _resolve_target_size(original_height, original_width, self.pn)[:2],
            )

        depth_np, normal_np = read_depth_normal_hypersim(
            str(depth_path),
            str(normal_path),
            _compute_hypersim_intrinsics(
                fov_value=float(record["settings_camera_fov"]),
                pixel_height=original_height,
                pixel_width=original_width,
            ),
            metric_scale=1.0,
        )

        # Some raw Hypersim normal maps contain invalid values. Zero them out before
        # resizing so interpolation and VAE encoding stay finite, and exclude them from
        # the supervision mask.
        valid_normal_np = np.isfinite(normal_np).all(axis=2) & (np.linalg.norm(normal_np, axis=2) > 1e-6)
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        normal_np = np.nan_to_num(normal_np, nan=0.0, posinf=0.0, neginf=0.0)

        depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(0)
        normal_tensor = torch.from_numpy(normal_np).float().permute(2, 0, 1)
        valid_normal_tensor = torch.from_numpy(valid_normal_np.astype(np.float32)).unsqueeze(0)
        target_height, target_width, template = _resolve_target_size(depth_np.shape[0], depth_np.shape[1], self.pn)
        target_hw = (target_height, target_width)
        depth_tensor = _resize_tensor(depth_tensor, target_hw, mode="nearest")
        normal_tensor = _resize_tensor(normal_tensor, target_hw, mode="bilinear", normalize_normals=True)
        valid_normal_tensor = _resize_tensor(valid_normal_tensor, target_hw, mode="nearest") > 0.5
        mask = (depth_tensor > 0) & valid_normal_tensor.bool()

        metadata = {
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "normal_path": str(normal_path),
            "partition": self.partition,
            "index": index,
            "stem": image_path.stem,
            "h_div_w": float(depth_np.shape[0]) / float(depth_np.shape[1]),
            "h_div_w_template": template,
            "target_size": target_hw,
            "original_size": (original_height, original_width),
        }
        return {
            "image": image_tensor.clamp(0.0, 1.0),
            "target": normal_tensor.clamp(-1.0, 1.0),
            "mask": mask.bool(),
            "metadata": metadata,
        }


class NYUv2ParquetNormalDataset(Dataset):
    """Read tanganke/nyuv2 Hugging Face parquet shards for RGB-to-normal evaluation."""

    def __init__(
        self,
        root: str,
        partition: str = "val",
        pn: str = "0.06M",
        max_samples: int = 0,
    ) -> None:
        super().__init__()
        if partition not in {"train", "val"}:
            raise ValueError(f"partition must be one of train/val for NYUv2 parquet, got {partition}")
        if pn not in {"0.06M", "0.25M", "1M"}:
            raise ValueError(f"Unsupported pn: {pn}")

        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise ImportError("NYUv2ParquetNormalDataset requires pyarrow. Install it with `pip install pyarrow`.") from exc

        self.root = Path(root)
        self.partition = partition
        self.pn = pn
        data_dir = self.root / "data"
        if not data_dir.is_dir():
            data_dir = self.root
        self.files = sorted(data_dir.glob(f"{partition}-*.parquet"))
        if not self.files:
            raise FileNotFoundError(f"No {partition}-*.parquet files found under {data_dir}")

        import pyarrow.parquet as pq

        self.records: list[tuple[int, int]] = []
        for file_index, path in enumerate(self.files):
            metadata = pq.ParquetFile(path).metadata
            for row_index in range(metadata.num_rows):
                self.records.append((file_index, row_index))
        if max_samples > 0:
            self.records = self.records[:max_samples]
        self._table_cache: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_table(self, file_index: int) -> Any:
        table = self._table_cache.get(file_index)
        if table is None:
            import pyarrow.parquet as pq

            table = pq.read_table(self.files[file_index], columns=["image", "normal", "depth"])
            self._table_cache.clear()
            self._table_cache[file_index] = table
        return table

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, row_index = self.records[index]
        table = self._load_table(file_index)
        image_tensor = torch.from_numpy(np.asarray(table["image"][row_index].as_py(), dtype=np.float32))
        normal_tensor = torch.from_numpy(np.asarray(table["normal"][row_index].as_py(), dtype=np.float32))
        depth_tensor = torch.from_numpy(np.asarray(table["depth"][row_index].as_py(), dtype=np.float32))

        normal_norm = torch.linalg.norm(normal_tensor, dim=0, keepdim=True)
        valid_normal = torch.isfinite(normal_tensor).all(dim=0, keepdim=True) & (normal_norm > 1e-6)
        normal_tensor = torch.nan_to_num(normal_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        normal_tensor = normal_tensor / torch.linalg.norm(normal_tensor, dim=0, keepdim=True).clamp_min(1e-6)
        depth_tensor = torch.nan_to_num(depth_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        target_height, target_width, template = _resolve_target_size(normal_tensor.shape[-2], normal_tensor.shape[-1], self.pn)
        target_hw = (target_height, target_width)
        image_tensor = _resize_tensor(image_tensor, target_hw, mode="bilinear").clamp(0.0, 1.0)
        normal_tensor = _resize_tensor(normal_tensor, target_hw, mode="bilinear", normalize_normals=True).clamp(-1.0, 1.0)
        depth_tensor = _resize_tensor(depth_tensor, target_hw, mode="nearest")
        valid_normal = _resize_tensor(valid_normal.float(), target_hw, mode="nearest") > 0.5
        mask = (depth_tensor > 0) & valid_normal.bool()

        metadata = {
            "parquet_path": str(self.files[file_index]),
            "partition": self.partition,
            "index": index,
            "row_index": row_index,
            "stem": f"{self.partition}_{index:06d}",
            "h_div_w": float(288) / float(384),
            "h_div_w_template": template,
            "target_size": target_hw,
            "original_size": (288, 384),
        }
        return {
            "image": image_tensor,
            "target": normal_tensor,
            "mask": mask.bool(),
            "metadata": metadata,
        }


def collate_normal_estimation_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "mask": torch.stack([sample["mask"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }
