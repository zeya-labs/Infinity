from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

from .file_io import read_depth_normal_hypersim, read_hdf5


ROOT_DIR = Path(__file__).resolve().parents[2]
DSINE_EVAL_DATASETS = {
    "scannet": {"official_split": "test", "png_normal": True},
    "ibims": {"official_split": "ibims", "png_normal": False},
    "sintel": {"official_split": "sintel", "png_normal": False},
}


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


def load_hypersim_normal_sample_from_metadata(metadata: dict[str, Any], pn: str) -> dict[str, Any]:
    image_path = Path(metadata["image_path"])
    depth_path = Path(metadata["depth_path"])
    normal_path = Path(metadata["normal_path"])
    fov_value = float(metadata["settings_camera_fov"])

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        original_width, original_height = image.size
        image_tensor = _resize_image(
            image,
            _resolve_target_size(original_height, original_width, pn)[:2],
        )

    depth_np, normal_np = read_depth_normal_hypersim(
        str(depth_path),
        str(normal_path),
        _compute_hypersim_intrinsics(
            fov_value=fov_value,
            pixel_height=original_height,
            pixel_width=original_width,
        ),
        metric_scale=1.0,
    )

    valid_normal_np = np.isfinite(normal_np).all(axis=2) & (np.linalg.norm(normal_np, axis=2) > 1e-6)
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    normal_np = np.nan_to_num(normal_np, nan=0.0, posinf=0.0, neginf=0.0)

    depth_tensor = torch.from_numpy(depth_np).float().unsqueeze(0)
    normal_tensor = torch.from_numpy(normal_np).float().permute(2, 0, 1)
    valid_normal_tensor = torch.from_numpy(valid_normal_np.astype(np.float32)).unsqueeze(0)
    target_height, target_width, template = _resolve_target_size(depth_np.shape[0], depth_np.shape[1], pn)
    target_hw = (target_height, target_width)
    depth_tensor = _resize_tensor(depth_tensor, target_hw, mode="nearest")
    normal_tensor = _resize_tensor(normal_tensor, target_hw, mode="bilinear", normalize_normals=True)
    valid_normal_tensor = _resize_tensor(valid_normal_tensor, target_hw, mode="nearest") > 0.5
    mask = (depth_tensor > 0) & valid_normal_tensor.bool()

    sample_metadata = dict(metadata)
    sample_metadata.update(
        {
            "h_div_w": float(depth_np.shape[0]) / float(depth_np.shape[1]),
            "h_div_w_template": template,
            "target_size": [int(target_hw[0]), int(target_hw[1])],
            "original_size": (original_height, original_width),
        }
    )
    return {
        "image": image_tensor.clamp(0.0, 1.0),
        "target": normal_tensor.clamp(-1.0, 1.0),
        "mask": mask.bool(),
        "metadata": sample_metadata,
    }


def _resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.extend([Path.cwd() / path, ROOT_DIR / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else path


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_dsine_eval_exr(path: Path) -> np.ndarray:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2

        normal_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if normal_bgr is None:
            raise ValueError(f"OpenCV could not read EXR normal: {path}")
        return cv2.cvtColor(normal_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    except Exception as cv2_exc:
        try:
            import imageio.v3 as iio

            normal = np.asarray(iio.imread(path), dtype=np.float32)
        except Exception as imageio_exc:
            raise RuntimeError(
                f"Failed to read EXR normal {path}. Install OpenCV with EXR support "
                "or an imageio EXR backend."
            ) from imageio_exc
        if normal.ndim != 3 or normal.shape[2] < 3:
            raise ValueError(f"EXR normal must have shape HxWx3, got {normal.shape} from {path}") from cv2_exc
        return normal[:, :, :3]


def _read_dsine_eval_normal(path: Path, *, png_normal: bool) -> tuple[np.ndarray, np.ndarray]:
    if png_normal:
        normal_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        valid = normal_rgb.sum(axis=2) > 0
        normal = (normal_rgb / 255.0) * 2.0 - 1.0
    else:
        normal = _read_dsine_eval_exr(path)
        valid = np.linalg.norm(normal, axis=2) > 0.5
    normal = np.nan_to_num(normal.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = valid & np.isfinite(normal).all(axis=2) & (np.linalg.norm(normal, axis=2) > 1e-6)
    return normal, valid.astype(np.float32)


def _replace_img_suffix(path: str, replacement: str) -> str:
    stem, ext = os.path.splitext(path)
    if stem.endswith("_img"):
        return stem[:-4] + replacement
    return stem + replacement


def load_vkitti2_normal_sample_from_metadata(metadata: dict[str, Any], pn: str) -> dict[str, Any]:
    base_dir = Path(metadata["manifest_dir"]) if metadata.get("manifest_dir") else None
    image_path = _resolve_path(metadata["image_path"], base_dir)
    normal_path = _resolve_path(metadata["normal_path"], base_dir)
    mask_path = _resolve_path(metadata["mask_path"], base_dir)

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        original_width, original_height = image.size
        target_height, target_width, template = _resolve_target_size(original_height, original_width, pn)
        target_hw = (target_height, target_width)
        image_tensor = _resize_image(image, target_hw)

    normal_np = np.load(normal_path).astype(np.float32)
    if normal_np.ndim != 3 or normal_np.shape[2] != 3:
        raise ValueError(f"VKITTI2 normal must have shape HxWx3, got {normal_np.shape} from {normal_path}")
    # Existing VKITTI2 preprocessing saves D2NT normals with flipped=true, matching the
    # final Hypersim training convention (x left, y up, z backward). If an older
    # manifest explicitly says otherwise, flip it here into the same convention.
    if not _truthy(metadata.get("flipped"), default=True):
        normal_np *= -1.0

    with Image.open(mask_path) as mask_handle:
        valid_np = np.asarray(mask_handle.convert("L")) > 0
    valid_np = valid_np & np.isfinite(normal_np).all(axis=2) & (np.linalg.norm(normal_np, axis=2) > 1e-6)
    normal_np = np.nan_to_num(normal_np, nan=0.0, posinf=0.0, neginf=0.0)

    normal_tensor = torch.from_numpy(normal_np).float().permute(2, 0, 1)
    valid_normal_tensor = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
    normal_tensor = _resize_tensor(normal_tensor, target_hw, mode="bilinear", normalize_normals=True)
    valid_normal_tensor = _resize_tensor(valid_normal_tensor, target_hw, mode="nearest") > 0.5

    sample_metadata = dict(metadata)
    sample_metadata.update(
        {
            "target_size": [int(target_hw[0]), int(target_hw[1])],
            "original_size": (original_height, original_width),
            "h_div_w": float(original_height) / float(original_width),
            "h_div_w_template": template,
        }
    )
    return {
        "image": image_tensor.clamp(0.0, 1.0),
        "target": normal_tensor.clamp(-1.0, 1.0),
        "mask": valid_normal_tensor.bool(),
        "metadata": sample_metadata,
    }


def load_dsine_eval_normal_sample_from_metadata(metadata: dict[str, Any], pn: str) -> dict[str, Any]:
    dataset_name = str(metadata["dataset"]).lower()
    dataset_config = DSINE_EVAL_DATASETS[dataset_name]
    base_dir = Path(metadata["dataset_root"])
    image_path = _resolve_path(metadata["image_path"], base_dir)
    normal_path = _resolve_path(metadata["normal_path"], base_dir)

    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        original_width, original_height = image.size
        target_height, target_width, template = _resolve_target_size(original_height, original_width, pn)
        target_hw = (target_height, target_width)
        image_tensor = _resize_image(image, target_hw)

    normal_np, valid_np = _read_dsine_eval_normal(normal_path, png_normal=bool(dataset_config["png_normal"]))
    if normal_np.ndim != 3 or normal_np.shape[2] != 3:
        raise ValueError(f"{dataset_name} normal must have shape HxWx3, got {normal_np.shape} from {normal_path}")
    normal_tensor = torch.from_numpy(normal_np).float().permute(2, 0, 1)
    valid_normal_tensor = torch.from_numpy(valid_np.astype(np.float32)).unsqueeze(0)
    normal_tensor = _resize_tensor(normal_tensor, target_hw, mode="bilinear", normalize_normals=True)
    valid_normal_tensor = _resize_tensor(valid_normal_tensor, target_hw, mode="nearest") > 0.5

    sample_metadata = dict(metadata)
    sample_metadata.update(
        {
            "target_size": [int(target_hw[0]), int(target_hw[1])],
            "original_size": (original_height, original_width),
            "h_div_w": float(original_height) / float(original_width),
            "h_div_w_template": template,
        }
    )
    return {
        "image": image_tensor.clamp(0.0, 1.0),
        "target": normal_tensor.clamp(-1.0, 1.0),
        "mask": valid_normal_tensor.bool(),
        "metadata": sample_metadata,
    }


def load_normal_sample_from_metadata(metadata: dict[str, Any], pn: str) -> dict[str, Any]:
    dataset_name = str(metadata.get("dataset", "hypersim")).lower()
    if dataset_name == "hypersim":
        return load_hypersim_normal_sample_from_metadata(metadata, pn)
    if dataset_name == "vkitti2":
        return load_vkitti2_normal_sample_from_metadata(metadata, pn)
    if dataset_name in DSINE_EVAL_DATASETS:
        return load_dsine_eval_normal_sample_from_metadata(metadata, pn)
    raise ValueError(f"Unsupported normal dataset in metadata: {dataset_name}")


def _load_jsonl_manifest(manifest_path: Path, required_fields: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record in {manifest_path} at line {line_number} must be a JSON object")
            missing = [field for field in required_fields if field not in record]
            if missing:
                raise ValueError(
                    f"Manifest record in {manifest_path} at line {line_number} "
                    f"missing required fields: {missing}"
                )
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return records


class HypersimNormalDataset(Dataset):
    """Read raw Hypersim jpg+hdf5 samples for RGB-to-normal estimation."""

    def __init__(
        self,
        root: str,
        partition: str = "train",
        pn: str = "0.06M",
        max_samples: int = 0,
        metadata_only: bool = False,
        filter_depth_nan: bool = False,
    ) -> None:
        super().__init__()
        if partition not in {"train", "val", "test"}:
            raise ValueError(f"partition must be one of train/val/test, got {partition}")

        self.root = Path(root)
        self.metadata_only = metadata_only
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
        if filter_depth_nan:
            self.records = self._records_without_depth_nan(csv_path)
        if max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def _records_without_depth_nan(self, csv_path: Path) -> list[dict[str, str]]:
        cache_path = csv_path.with_suffix(".no_depth_nan.jsonl")
        if cache_path.is_file():
            records = []
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
            return records

        records = []
        for record in self.records:
            depth_path = self.root / record["depth"]
            depth = read_hdf5(str(depth_path))
            if np.isnan(depth).any():
                continue
            records.append(record)
        with cache_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return records

    def _metadata_for_record(
        self,
        index: int,
        record: dict[str, str],
        image_path: Path,
        depth_path: Path,
        normal_path: Path,
        target_hw: tuple[int, int],
        original_hw: tuple[int, int] | None = None,
        h_div_w: float | None = None,
        template: float | None = None,
    ) -> dict[str, Any]:
        if template is None:
            template = _nearest_h_div_w_template(*(original_hw or target_hw))
        if h_div_w is None:
            height, width = original_hw or target_hw
            h_div_w = float(height) / float(width)
        original_size = original_hw if original_hw is not None else target_hw
        return {
            "dataset": "hypersim",
            "image_path": str(image_path),
            "depth_path": str(depth_path),
            "normal_path": str(normal_path),
            "partition": self.partition,
            "index": index,
            "stem": image_path.stem,
            "h_div_w": h_div_w,
            "h_div_w_template": template,
            "target_size": [int(target_hw[0]), int(target_hw[1])],
            "original_size": original_size,
            "settings_camera_fov": float(record["settings_camera_fov"]),
        }

    def _metadata_only_sample(self, index: int) -> dict[str, Any]:
        metadata = self.get_metadata(index)
        return {
            "image": torch.empty(0),
            "target": torch.empty(0),
            "mask": torch.empty(0, dtype=torch.bool),
            "metadata": metadata,
        }

    def get_metadata(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_index = int(record.get("__original_index", index))
        image_path = self.root / record["images"]
        depth_path = self.root / record["depth"]
        normal_path = self.root / record["normal"]
        return self._metadata_for_record(
            source_index,
            record,
            image_path,
            depth_path,
            normal_path,
            _resolve_target_size(768, 1024, self.pn)[:2],
            original_hw=(768, 1024),
        )

    def load_full_sample(self, index: int) -> dict[str, Any]:
        metadata = self.get_metadata(index)
        return load_hypersim_normal_sample_from_metadata(metadata, self.pn)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.metadata_only:
            return self._metadata_only_sample(index)
        return self.load_full_sample(index)


class VKITTI2NormalDataset(Dataset):
    """Read preprocessed VKITTI2 RGB + D2NT normal samples for RGB-to-normal training."""

    def __init__(
        self,
        root: str,
        partition: str = "train",
        pn: str = "0.06M",
        max_samples: int = 0,
        metadata_only: bool = False,
    ) -> None:
        super().__init__()
        if pn not in {"0.06M", "0.25M", "1M"}:
            raise ValueError(f"Unsupported pn: {pn}")

        self.root = Path(root)
        self.partition = partition
        self.pn = pn
        self.metadata_only = metadata_only
        self.manifest_path = self._resolve_manifest_path(self.root)
        self.manifest_dir = self.manifest_path.parent
        self.records = _load_jsonl_manifest(self.manifest_path, required_fields=("rgb_path", "normal_path", "mask_path"))
        if max_samples > 0:
            self.records = self.records[:max_samples]
        self.original_hw = self._infer_original_hw()

    @staticmethod
    def _resolve_manifest_path(root: Path) -> Path:
        candidates = [
            root / "manifest.jsonl",
            root / "processed" / "normals_lotus_svd" / "manifest.jsonl",
            root / "processed" / "normals_d2nt_v3" / "manifest.jsonl",
            root / "processed" / "normals_from_depth" / "manifest.jsonl",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"VKITTI2 manifest.jsonl not found under {root}")

    def __len__(self) -> int:
        return len(self.records)

    def _infer_original_hw(self) -> tuple[int, int]:
        if not self.records:
            return (375, 1242)
        image_path = _resolve_path(self.records[0]["rgb_path"], self.manifest_dir)
        with Image.open(image_path) as image_handle:
            original_width, original_height = image_handle.size
        return int(original_height), int(original_width)

    def _metadata_for_record(self, index: int, record: dict[str, Any]) -> dict[str, Any]:
        original_height, original_width = self.original_hw
        target_height, target_width, template = _resolve_target_size(original_height, original_width, self.pn)
        metadata = dict(record)
        metadata.update(
            {
                "dataset": "vkitti2",
                "image_path": record["rgb_path"],
                "partition": self.partition,
                "index": int(record.get("__original_index", index)),
                "stem": f"{record.get('scene', 'scene')}_{record.get('variant', 'variant')}_{record.get('camera', 'camera')}_{int(record.get('frame', index)):05d}",
                "h_div_w": float(original_height) / float(original_width),
                "h_div_w_template": template,
                "target_size": [int(target_height), int(target_width)],
                "original_size": (original_height, original_width),
                "manifest_dir": str(self.manifest_dir),
            }
        )
        return metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.get_metadata(index)
        if self.metadata_only:
            return {
                "image": torch.empty(0),
                "target": torch.empty(0),
                "mask": torch.empty(0, dtype=torch.bool),
                "metadata": metadata,
            }
        return load_vkitti2_normal_sample_from_metadata(metadata, self.pn)

    def get_metadata(self, index: int) -> dict[str, Any]:
        return self._metadata_for_record(index, self.records[index])


class DSINEEvalNormalDataset(Dataset):
    """Read DSINE evaluation-package RGB + GT normal samples."""

    def __init__(
        self,
        root: str,
        dataset: str,
        partition: str = "test",
        pn: str = "0.06M",
        max_samples: int = 0,
        metadata_only: bool = False,
    ) -> None:
        super().__init__()
        dataset = dataset.lower()
        if dataset not in DSINE_EVAL_DATASETS:
            raise ValueError(f"dataset must be one of {sorted(DSINE_EVAL_DATASETS)}, got {dataset}")
        if pn not in {"0.06M", "0.25M", "1M"}:
            raise ValueError(f"Unsupported pn: {pn}")

        self.root = Path(root)
        self.dataset = dataset
        self.partition = partition
        self.pn = pn
        self.metadata_only = metadata_only
        self.dataset_root = self._resolve_dataset_root(self.root, dataset)
        self.records = self._load_records(partition)
        if max_samples > 0:
            self.records = self.records[:max_samples]
        if not self.records:
            raise FileNotFoundError(
                f"No DSINE eval samples found for {dataset} under {self.dataset_root}. "
                "Run scripts/download-data/dsine-eval-tos-bundle/run_download_from_tos.ps1 first."
            )

    @staticmethod
    def _resolve_dataset_root(root: Path, dataset: str) -> Path:
        candidates = [
            root if root.name.lower() == dataset else root / dataset,
            root / "dsine_eval" / dataset,
            root,
        ]
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.rglob("*_img.*")):
                return candidate
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return candidates[0]

    def _split_candidates(self, partition: str) -> list[Path]:
        official_split = str(DSINE_EVAL_DATASETS[self.dataset]["official_split"])
        names = []
        for name in (partition, official_split):
            if name and name not in names:
                names.append(name)
        candidates: list[Path] = []
        for name in names:
            candidates.extend(
                [
                    self.dataset_root / "split" / f"{name}.txt",
                    self.dataset_root / f"{name}.txt",
                    ROOT_DIR
                    / "external"
                    / "normal_baselines"
                    / "repos"
                    / "DSINE"
                    / "data"
                    / "datasets"
                    / self.dataset
                    / "split"
                    / f"{name}.txt",
                ]
            )
        return candidates

    def _load_records(self, partition: str) -> list[str]:
        for split_path in self._split_candidates(partition):
            if not split_path.is_file():
                continue
            records = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if records and (self.dataset_root / records[0]).is_file():
                return records
        return sorted(path.relative_to(self.dataset_root).as_posix() for path in self.dataset_root.rglob("*_img.*"))

    def __len__(self) -> int:
        return len(self.records)

    def _metadata_for_record(self, index: int, record: str) -> dict[str, Any]:
        normal_ext = ".png" if bool(DSINE_EVAL_DATASETS[self.dataset]["png_normal"]) else ".exr"
        normal_path = self.dataset_root / _replace_img_suffix(record, f"_normal{normal_ext}")
        relative_stem = _replace_img_suffix(record, "")
        return {
            "dataset": self.dataset,
            "dataset_root": str(self.dataset_root),
            "image_path": record,
            "normal_path": normal_path.relative_to(self.dataset_root).as_posix(),
            "partition": self.partition,
            "index": index,
            "stem": relative_stem.replace("/", "_"),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        metadata = self.get_metadata(index)
        if self.metadata_only:
            return {
                "image": torch.empty(0),
                "target": torch.empty(0),
                "mask": torch.empty(0, dtype=torch.bool),
                "metadata": metadata,
            }
        return load_dsine_eval_normal_sample_from_metadata(metadata, self.pn)

    def get_metadata(self, index: int) -> dict[str, Any]:
        return self._metadata_for_record(index, self.records[index])


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
    if all(sample["image"].numel() == 0 for sample in samples):
        return {
            "image": torch.empty(0),
            "target": torch.empty(0),
            "mask": torch.empty(0, dtype=torch.bool),
            "metadata": [sample["metadata"] for sample in samples],
        }
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "target": torch.stack([sample["target"] for sample in samples], dim=0),
        "mask": torch.stack([sample["mask"] for sample in samples], dim=0),
        "metadata": [sample["metadata"] for sample in samples],
    }
