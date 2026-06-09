#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYTHON = ROOT / ".venv" / "bin" / "python"
BASELINE_RUNNER = ROOT / "external" / "normal_baselines" / "scripts" / "run_normal_baseline_compare.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_EXPORT_DATASET = None
_EXPORT_WORK_DIR: Path | None = None
_EVAL_METHOD_DIR: Path | None = None
_EVAL_METHOD: str | None = None
_EVAL_DATASET: str | None = None


@dataclass(frozen=True)
class NormalConvention:
    name: str
    to_eval_perm: tuple[int, int, int]
    to_eval_signs: tuple[int, int, int]
    description: str


EVAL_CONVENTION = "eval_camera_xyz"
IDENTITY_CONVENTION = NormalConvention(
    name=EVAL_CONVENTION,
    to_eval_perm=(0, 1, 2),
    to_eval_signs=(1, 1, 1),
    description="Canonical metric convention used by this script after dataset export.",
)
DSINE_RIGHT_DOWN_FRONT_CONVENTION = NormalConvention(
    name="dsine_right_down_front_outward",
    to_eval_perm=(0, 2, 1),
    to_eval_signs=(-1, -1, 1),
    description="DSINE eval package normals. Eval mapping: (x, y, z) -> (-x, -z, y).",
)
GT_EXPORT_DATASETS = ("nyuv2", "hypersim", "scannet", "ibims", "sintel")

# Dataset readers/exporters normalize each dataset into the canonical evaluation
# convention before writing _eval_set/gt/*.npy. Keep this table explicit so adding
# a dataset means declaring its target convention in one place.
DATASET_TARGET_CONVENTIONS: dict[str, NormalConvention] = {
    "nyuv2": IDENTITY_CONVENTION,
    "hypersim": IDENTITY_CONVENTION,
    "scannet": DSINE_RIGHT_DOWN_FRONT_CONVENTION,
    "ibims": DSINE_RIGHT_DOWN_FRONT_CONVENTION,
    "sintel": DSINE_RIGHT_DOWN_FRONT_CONVENTION,
}

# Model-output conventions observed from each official adapter's raw *.npy output.
# Values transform model output into EVAL_CONVENTION: out_eval[c] =
# signs[c] * out_model[perm[c]].
MODEL_OUTPUT_CONVENTIONS: dict[str, NormalConvention] = {
    "ours": NormalConvention(
        name="ours_model_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Infinity normal output already matches the exported eval convention.",
    ),
    "marigold": NormalConvention(
        name="marigold_normals_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(1, -1, 1),
        description="Marigold normals_npy output. Eval mapping: (x, y, z) -> (x, -z, y).",
    ),
    "geowizard": NormalConvention(
        name="geowizard_normal_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="GeoWizard normal_npy output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
    "stablenormal": NormalConvention(
        name="stablenormal_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="StableNormal output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
    "lotusg": NormalConvention(
        name="lotus_g_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="Lotus-G output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
    "lotusd": NormalConvention(
        name="lotus_d_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="Lotus-D output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
    "dsine": NormalConvention(
        name="dsine_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="DSINE output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
    "metric3dv2": NormalConvention(
        name="metric3d_v2_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(1, 1, -1),
        description="Metric3D v2 output. Eval mapping: (x, y, z) -> (x, z, -y).",
    ),
    "omnidata_v2": NormalConvention(
        name="omnidata_v2_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Omnidata V2 wrapper saves normals in the exported eval convention.",
    ),
    "marigold_e2eft": NormalConvention(
        name="marigold_e2eft_xyz",
        to_eval_perm=(0, 2, 1),
        to_eval_signs=(-1, -1, 1),
        description="Marigold E2E-FT output. Eval mapping: (x, y, z) -> (-x, -z, y).",
    ),
}

# Same normal-map visualization convention as infinity.normal_estimation.normals_to_vis.
DISPLAY_FROM_EVAL_PERM = (0, 1, 2)
DISPLAY_FROM_EVAL_SIGNS = (-1, 1, 1)


def parse_methods(raw: list[str]) -> list[str]:
    methods: list[str] = []
    for value in raw:
        methods.extend(item.strip().lower() for item in value.replace(",", " ").split() if item.strip())
    ordered: list[str] = []
    seen: set[str] = set()
    for method in methods:
        if method not in seen:
            seen.add(method)
            ordered.append(method)
    return ordered


def clean_env(cuda_device: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_device
    return env


def no_proxy_env() -> dict[str, str]:
    env = clean_env()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    return env


def visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return [str(index) for index in range(8)]
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if not devices or devices == ["-1"]:
        return []
    return devices


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def dataset_default_root(dataset: str) -> Path:
    if dataset == "toy":
        return ROOT / "data" / "infinity_toy_data"
    if dataset == "nyuv2":
        return ROOT / "data" / "NYUv2" / "hf-parquet" / "tanganke" / "nyuv2" / "data"
    if dataset == "hypersim":
        return ROOT / "data" / "hypersim" / "processed" / "hypersim"
    if dataset in {"scannet", "ibims", "sintel"}:
        return ROOT / "data" / "dsine_eval"
    raise ValueError(f"Unsupported dataset: {dataset}")


def resolve_image_dir(dataset: str, data_root: Path) -> Path:
    if dataset == "toy":
        return data_root / "images" if (data_root / "images").is_dir() else data_root
    return data_root


def resolve_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    if input_dir.is_file():
        return [input_dir]
    return sorted(path for path in input_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def prepare_input_shards(input_dir: Path, shard_root: Path, shard_count: int) -> list[Path]:
    image_paths = resolve_image_paths(input_dir)
    if shard_count <= 1 or len(image_paths) <= 1:
        return [input_dir]
    shard_count = min(shard_count, len(image_paths))
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True, exist_ok=True)
    shards = [shard_root / f"shard_{index:02d}" for index in range(shard_count)]
    for shard in shards:
        shard.mkdir(parents=True, exist_ok=True)
    for index, image_path in enumerate(image_paths):
        target = shards[index % shard_count] / image_path.name
        target.symlink_to(image_path.resolve())
    return shards


def merge_shard_outputs(shard_outputs: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_rows: list[dict[str, object]] = []
    for shard_output in shard_outputs:
        if not shard_output.exists():
            continue
        timing_path = shard_output / "inference_times.json"
        if timing_path.exists():
            payload = json.loads(timing_path.read_text(encoding="utf-8"))
            rows = payload.get("images") if isinstance(payload, dict) else payload
            if isinstance(rows, list):
                timing_rows.extend(row for row in rows if isinstance(row, dict))
        for path in shard_output.rglob("*"):
            if not path.is_file() or path.name in {"metrics.json", "inference_times.json"}:
                continue
            relative = path.relative_to(shard_output)
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.unlink()
            shutil.copy2(path, target)
    if timing_rows:
        (output_dir / "inference_times.json").write_text(
            json.dumps({"images": timing_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def save_image_tensor(image_chw: torch.Tensor, path: Path) -> None:
    array = image_chw.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).mul(255).byte().numpy()
    Image.fromarray(array).save(path)


def parse_worker_count(value: str, sample_count: int) -> int:
    if value.lower() == "auto":
        return max(1, min(sample_count, os.cpu_count() or 1))
    parsed = int(value)
    if parsed <= 0:
        return max(1, min(sample_count, os.cpu_count() or 1))
    return max(1, min(sample_count, parsed))


def sanitize_sample_id(value: str) -> str:
    sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return sample_id or "sample"


def sample_id_from_metadata(index: int, metadata: dict[str, Any]) -> str:
    dataset = str(metadata.get("dataset", "")).lower()
    if dataset == "hypersim" and metadata.get("image_path"):
        image_path = Path(str(metadata["image_path"]))
        scene = image_path.parents[2].name if len(image_path.parents) >= 3 else ""
        camera = image_path.parent.name
        return sanitize_sample_id("_".join(part for part in (scene, camera, image_path.stem) if part))
    return sanitize_sample_id(str(metadata.get("stem") or f"{index:08d}"))


def init_export_worker(dataset: str, data_root: str, partition: str, pn: str, max_samples: int, work_dir: str) -> None:
    global _EXPORT_DATASET, _EXPORT_WORK_DIR
    torch.set_num_threads(1)
    from infinity.normal_estimation import DSINEEvalNormalDataset, HypersimNormalDataset, NYUv2ParquetNormalDataset

    if dataset == "hypersim":
        _EXPORT_DATASET = HypersimNormalDataset(root=data_root, partition=partition, pn=pn, max_samples=max_samples)
    elif dataset == "nyuv2":
        _EXPORT_DATASET = NYUv2ParquetNormalDataset(root=data_root, partition=partition, pn=pn, max_samples=max_samples)
    elif dataset in {"scannet", "ibims", "sintel"}:
        _EXPORT_DATASET = DSINEEvalNormalDataset(
            root=data_root,
            dataset=dataset,
            partition=partition,
            pn=pn,
            max_samples=max_samples,
        )
    else:
        raise ValueError(f"Cannot export GT dataset for {dataset}")
    _EXPORT_WORK_DIR = Path(work_dir)


def export_one_sample(index: int) -> dict[str, object]:
    if _EXPORT_DATASET is None or _EXPORT_WORK_DIR is None:
        raise RuntimeError("Export worker is not initialized.")
    sample = _EXPORT_DATASET[index]
    metadata = dict(sample["metadata"])
    sample_id = sample_id_from_metadata(index, metadata)
    image_path = _EXPORT_WORK_DIR / "images" / f"{sample_id}.png"
    target_path = _EXPORT_WORK_DIR / "gt" / f"{sample_id}_normal.npy"
    mask_path = _EXPORT_WORK_DIR / "mask" / f"{sample_id}_mask.png"
    save_image_tensor(sample["image"], image_path)
    np.save(target_path, sample["target"].permute(1, 2, 0).cpu().numpy().astype(np.float32))
    Image.fromarray(sample["mask"].squeeze(0).cpu().numpy().astype(np.uint8) * 255).save(mask_path)
    return {
        "id": sample_id,
        "image": str(image_path),
        "target": str(target_path),
        "mask": str(mask_path),
        "source_image": metadata.get("image_path", metadata.get("parquet_path", "")),
        "target_size": metadata.get("target_size", []),
    }


def load_eval_manifest(work_dir: Path) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    with (work_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def export_eval_set(
    *,
    dataset: str,
    data_root: Path,
    partition: str,
    pn: str,
    max_samples: int,
    work_dir: Path,
    workers: str,
) -> tuple[Path, list[dict[str, object]]]:
    manifest_path = work_dir / "manifest.jsonl"
    images_dir = work_dir / "images"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "gt").mkdir(parents=True, exist_ok=True)
    (work_dir / "mask").mkdir(parents=True, exist_ok=True)

    init_export_worker(dataset, str(data_root), partition, pn, max_samples, str(work_dir))
    sample_count = len(_EXPORT_DATASET)
    worker_count = parse_worker_count(workers, sample_count)
    print(json.dumps({"export_eval_set": str(work_dir), "samples": sample_count, "workers": worker_count}), flush=True)
    if worker_count == 1:
        manifest = [export_one_sample(index) for index in range(sample_count)]
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=init_export_worker,
            initargs=(dataset, str(data_root), partition, pn, max_samples, str(work_dir)),
        ) as pool:
            manifest = list(pool.imap(export_one_sample, range(sample_count), chunksize=8))
    manifest_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + ("\n" if manifest else ""),
        encoding="utf-8",
    )
    return images_dir, manifest


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool, env: dict[str, str], cuda_device: str | None = None) -> int:
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_device} " if cuda_device is not None else ""
    print("$ " + prefix + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    return int(subprocess.run([str(part) for part in cmd], cwd=str(cwd), env=env, check=False).returncode)


def ours_command(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> list[str]:
    cmd = [
        str(PYTHON),
        "tools/run_normal_estimation.py",
        "--model-path",
        str(args.ours_checkpoint),
        "--input-path",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--seed",
        args.ours_seed,
        "--top-k",
        args.ours_top_k,
        "--top-p",
        args.ours_top_p,
        "--tau",
        args.ours_tau,
        "--save-npy",
        "--timing-warmup",
        str(args.timing_warmup),
        "--timing-repeats",
        str(args.timing_repeats),
    ]
    if args.normal_tokenizer_ckpt:
        cmd.extend(["--normal-vae-ckpt", str(args.normal_tokenizer_ckpt)])
    if args.normal_vae_type:
        cmd.extend(["--normal-vae-type", str(args.normal_vae_type)])
    if args.ours_kv_cache_fast:
        cmd.append("--normal-kv-cache-fast")
    return cmd


def run_ours_sharded(args: argparse.Namespace, input_dir: Path, output_dir: Path, devices: list[str]) -> tuple[int, list[list[str]]]:
    shard_count = max(1, len(devices))
    input_shards = prepare_input_shards(input_dir, output_dir / "_input_shards", shard_count)
    if len(input_shards) == 1:
        cmd = ours_command(args, input_shards[0], output_dir)
        device = devices[0] if devices else None
        code = run_command(cmd, cwd=ROOT, dry_run=args.dry_run, env=clean_env(device), cuda_device=device)
        return code, [cmd]

    shard_output_root = output_dir / "_shards"
    if shard_output_root.exists():
        shutil.rmtree(shard_output_root)
    shard_output_root.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    processes: list[subprocess.Popen[bytes]] = []
    for index, shard_input in enumerate(input_shards):
        shard_output = shard_output_root / f"shard_{index:02d}"
        shard_output.mkdir(parents=True, exist_ok=True)
        cmd = ours_command(args, shard_input, shard_output)
        commands.append(cmd)
        device = devices[index % len(devices)] if devices else None
        print(f"$ CUDA_VISIBLE_DEVICES={device} " + " ".join(cmd), flush=True)
        if not args.dry_run:
            processes.append(subprocess.Popen(cmd, cwd=str(ROOT), env=clean_env(device)))
    codes = [process.wait() for process in processes] if not args.dry_run else [0 for _ in commands]
    if any(code != 0 for code in codes):
        return 2, commands
    if not args.dry_run:
        merge_shard_outputs([shard_output_root / f"shard_{index:02d}" for index in range(len(input_shards))], output_dir)
    return 0, commands


def timing_protocol() -> str:
    return (
        "Batch-size-1 per-image model inference latency measured inside each predictor with time.perf_counter and CUDA "
        "synchronization around the model-call section only. Each image uses unmeasured warmup iterations followed by "
        "measured repeats. Model loading, process startup, dataset export, image file I/O, metric computation, "
        "visualization writing, and result serialization are excluded."
    )


def load_inference_timing(method_dir: Path, method: str, enabled: bool) -> dict[str, object]:
    path = method_dir / "inference_times.json"
    unavailable = {
        "enabled": bool(enabled),
        "available": False,
        "method": method,
        "protocol": timing_protocol(),
        "reason": "No per-image inference_times.json was produced by this method wrapper.",
    }
    if not enabled:
        unavailable["reason"] = "Inference-time comparison disabled."
        return unavailable
    if not path.exists():
        return unavailable
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("images") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        unavailable["reason"] = f"Malformed timing file: {path}"
        return unavailable
    seconds: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        repeats = row.get("repeat_inference_seconds")
        if isinstance(repeats, list):
            seconds.extend(float(value) for value in repeats if isinstance(value, (int, float)))
        elif isinstance(row.get("inference_seconds"), (int, float)):
            seconds.append(float(row["inference_seconds"]))
    if not seconds:
        unavailable["reason"] = f"No valid per-image inference_seconds in {path}"
        return unavailable
    arr = np.asarray(seconds, dtype=np.float64)
    ci95 = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "enabled": True,
        "available": True,
        "method": method,
        "protocol": timing_protocol(),
        "timed_observations": int(arr.size),
        "images": int(len(rows)),
        "mean_seconds": float(arr.mean()),
        "median_seconds": float(np.median(arr)),
        "std_seconds": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "ci95_seconds": ci95,
        "min_seconds": float(arr.min()),
        "max_seconds": float(arr.max()),
        "images_per_second": float(1.0 / arr.mean()) if arr.mean() > 0 else None,
        "per_image_file": str(path),
    }


def update_metrics_timing(method_dir: Path, method: str, timing: dict[str, object]) -> None:
    path = method_dir / "metrics.json"
    payload: dict[str, object]
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {"method": method}
    payload["inference_time"] = timing
    metrics = payload.get("metrics")
    if isinstance(metrics, dict) and timing.get("available"):
        for key in ("mean_seconds", "median_seconds", "std_seconds", "images_per_second"):
            value = timing.get(key)
            if isinstance(value, (int, float)):
                metrics[f"inference_{key}"] = float(value)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_image_only_metrics(output_dir: Path, method: str, timing: dict[str, object] | None = None) -> None:
    payload = {
        "method": method,
        "num_png": len(list(output_dir.glob("*_normal.png"))),
        "num_npy": len(list(output_dir.glob("*_normal.npy"))),
        "metrics": {},
    }
    if timing is not None:
        payload["inference_time"] = timing
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_prediction(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        array = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0 * 2.0 - 1.0
    if array.ndim != 3:
        raise ValueError(f"Prediction must be HxWx3: {path} shape={array.shape}")
    if array.shape[2] == 3:
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()
    elif array.shape[0] == 3:
        tensor = torch.from_numpy(array).unsqueeze(0).float()
    else:
        raise ValueError(f"Prediction must be HxWx3 or 3xHxW: {path} shape={array.shape}")
    return tensor / torch.linalg.norm(tensor, dim=1, keepdim=True).clamp_min(1e-6)


def apply_signed_permutation_tensor(
    normal: torch.Tensor,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> torch.Tensor:
    transformed = normal[:, list(perm)].clone()
    sign_tensor = transformed.new_tensor(signs).view(1, 3, 1, 1)
    return transformed * sign_tensor


def apply_signed_permutation_array(
    normal: np.ndarray,
    perm: tuple[int, int, int],
    signs: tuple[int, int, int],
) -> np.ndarray:
    transformed = normal[:, :, list(perm)].copy()
    return transformed * np.asarray(signs, dtype=np.float32).reshape(1, 1, 3)


def normalize_prediction(prediction: torch.Tensor) -> torch.Tensor:
    return prediction / torch.linalg.norm(prediction, dim=1, keepdim=True).clamp_min(1e-6)


def dataset_target_convention(dataset: str) -> NormalConvention:
    if dataset not in DATASET_TARGET_CONVENTIONS:
        raise ValueError(f"No target normal convention registered for dataset={dataset!r}")
    return DATASET_TARGET_CONVENTIONS[dataset]


def model_output_convention(method: str) -> NormalConvention:
    if method not in MODEL_OUTPUT_CONVENTIONS:
        raise ValueError(f"No output normal convention registered for method={method!r}")
    return MODEL_OUTPUT_CONVENTIONS[method]


def convert_target_to_eval(target: torch.Tensor, dataset: str) -> torch.Tensor:
    convention = dataset_target_convention(dataset)
    return normalize_prediction(apply_signed_permutation_tensor(target, convention.to_eval_perm, convention.to_eval_signs))


def convert_prediction_to_eval(prediction: torch.Tensor, method: str) -> torch.Tensor:
    convention = model_output_convention(method)
    return normalize_prediction(apply_signed_permutation_tensor(prediction, convention.to_eval_perm, convention.to_eval_signs))


def eval_normal_to_display_rgb(normal_hwc: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    display_normal = apply_signed_permutation_array(normal_hwc, DISPLAY_FROM_EVAL_PERM, DISPLAY_FROM_EVAL_SIGNS)
    rgb = ((display_normal.clip(-1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    if mask is not None:
        rgb = rgb.copy()
        rgb[~mask] = (24, 24, 24)
    return rgb


def convention_payload(dataset: str, method: str) -> dict[str, object]:
    return {
        "canonical": EVAL_CONVENTION,
        "dataset_target": asdict(dataset_target_convention(dataset)),
        "model_output": asdict(model_output_convention(method)),
        "display_from_eval": {
            "perm": DISPLAY_FROM_EVAL_PERM,
            "signs": DISPLAY_FROM_EVAL_SIGNS,
            "description": "Normal-map display colors matching infinity.normal_estimation.normals_to_vis: eval (x, y, z) -> RGB (-x, y, z).",
        },
    }


def find_prediction_path(method_dir: Path, sample_id: str) -> Path | None:
    candidates = [
        method_dir / f"{sample_id}_normal.npy",
        method_dir / f"{sample_id}_normal.png",
        method_dir / f"{sample_id}_normals.npy",
        method_dir / f"{sample_id}_normals.png",
        method_dir / "normals_npy" / f"{sample_id}_normals.npy",
        method_dir / "normals_vis" / f"{sample_id}_normals.png",
        method_dir / "normal_npy" / f"{sample_id}_pred.npy",
        method_dir / "normal_vis" / f"{sample_id}_pred.png",
        method_dir / "normal_npy" / f"{sample_id}_normal.npy",
        method_dir / "normal_vis" / f"{sample_id}_normal.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(
        path
        for suffix in ("npy", "png")
        for path in method_dir.rglob(f"{sample_id}_normal*.{suffix}")
        if not any(part.startswith("_") for part in path.relative_to(method_dir).parts)
    )
    return matches[0] if matches else None


def init_eval_metric_worker(method_dir: str, method: str, dataset: str) -> None:
    global _EVAL_METHOD_DIR, _EVAL_METHOD, _EVAL_DATASET
    torch.set_num_threads(1)
    _EVAL_METHOD_DIR = Path(method_dir)
    _EVAL_METHOD = method
    _EVAL_DATASET = dataset


def evaluate_one_sample(item: dict[str, object]) -> dict[str, object]:
    from infinity.normal_estimation import compute_normal_metrics

    if _EVAL_METHOD_DIR is None or _EVAL_METHOD is None or _EVAL_DATASET is None:
        raise RuntimeError("Eval metric worker is not initialized.")
    sample_id = str(item["id"])
    pred_path = find_prediction_path(_EVAL_METHOD_DIR, sample_id)
    if pred_path is None:
        return {"id": sample_id, "missing": True}
    prediction = convert_prediction_to_eval(load_prediction(pred_path), _EVAL_METHOD)
    target = torch.from_numpy(np.load(str(item["target"])).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    target = convert_target_to_eval(target, _EVAL_DATASET)
    mask_np = np.asarray(Image.open(str(item["mask"])).convert("L")) > 127
    mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).bool()
    if prediction.shape[-2:] != target.shape[-2:]:
        prediction = torch.nn.functional.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
        prediction = normalize_prediction(prediction)
    _, metrics = compute_normal_metrics(
        prediction=prediction,
        target=target,
        mask=mask,
        l1_weight=0.0,
        angular_weight=0.0,
        latent_weight=0.0,
        norm_weight=0.0,
    )
    row = {key: float(value.item()) for key, value in metrics.items()}
    row["id"] = sample_id
    return row


def error_to_rgb(error_deg: np.ndarray, mask: np.ndarray, vmax: float = 45.0) -> np.ndarray:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.cm as cm

        rgb = (cm.turbo(np.clip(error_deg / vmax, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
    except Exception:
        scaled = np.clip(error_deg / vmax, 0.0, 1.0)
        rgb = np.stack(
            [
                (scaled * 255).astype(np.uint8),
                ((1.0 - np.abs(scaled - 0.5) * 2.0).clip(0.0, 1.0) * 255).astype(np.uint8),
                ((1.0 - scaled) * 255).astype(np.uint8),
            ],
            axis=2,
        )
    rgb = rgb.copy()
    rgb[~mask] = (24, 24, 24)
    return rgb


def error_legend_to_rgb(width: int, height: int, vmax: float = 45.0) -> np.ndarray:
    image = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    margin_x = 38
    bar_w = width - margin_x * 2
    bar_h = 42
    bar_y = height // 2 - 12
    values = np.tile(np.linspace(0.0, vmax, bar_w, dtype=np.float32), (bar_h, 1))
    legend = error_to_rgb(values, np.ones_like(values, dtype=bool), vmax=vmax)
    image.paste(Image.fromarray(legend), (margin_x, bar_y))
    draw.rectangle((margin_x, bar_y, margin_x + bar_w - 1, bar_y + bar_h - 1), outline=(220, 220, 220), width=1)

    draw.text((margin_x, bar_y - 34), "Angular error", fill=(235, 235, 235), font=font)
    for tick in (0.0, 11.25, 22.5, 33.75, vmax):
        x = margin_x + int(round((tick / vmax) * (bar_w - 1)))
        draw.line((x, bar_y + bar_h, x, bar_y + bar_h + 8), fill=(220, 220, 220), width=1)
        label = f"{tick:g}" if tick < vmax else f"{vmax:g}+"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (bbox[2] - bbox[0]) // 2, bar_y + bar_h + 12), label, fill=(220, 220, 220), font=small_font)
    draw.text((margin_x, height - 34), "degrees, clipped at 45", fill=(190, 190, 190), font=small_font)
    return np.asarray(image)


def write_representative_visualization(
    *,
    method_dir: Path,
    manifest: list[dict[str, object]],
    method: str,
    dataset: str,
    per_sample: list[dict[str, float | str]],
    max_rows: int = 6,
) -> None:
    if not per_sample:
        return
    by_id = {str(item["id"]): item for item in manifest}
    ordered = sorted(per_sample, key=lambda item: float(item["angle_deg"]))
    if len(ordered) <= max_rows:
        selected = ordered
    else:
        quantiles = np.linspace(0.05, 0.98, max_rows)
        selected = []
        used: set[int] = set()
        for quantile in quantiles:
            index = int(round(float(quantile) * (len(ordered) - 1)))
            while index in used and index + 1 < len(ordered):
                index += 1
            used.add(index)
            selected.append(ordered[index])

    cell_w, cell_h, label_h, cols = 320, 240, 34, 4
    canvas = Image.new("RGB", (cell_w * cols, len(selected) * (cell_h + label_h)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()

    for row_index, sample_metrics in enumerate(selected):
        sample_id = str(sample_metrics["id"])
        item = by_id[sample_id]
        pred_path = find_prediction_path(method_dir, sample_id)
        if pred_path is None:
            continue
        prediction = convert_prediction_to_eval(load_prediction(pred_path), method)
        target = torch.from_numpy(np.load(str(item["target"])).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        target = convert_target_to_eval(target, dataset)
        mask_np = np.asarray(Image.open(str(item["mask"])).convert("L")) > 127
        mask = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).bool()
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = torch.nn.functional.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
            prediction = normalize_prediction(prediction)

        prediction_hwc = prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        target_hwc = target[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        error_deg = np.degrees(np.arccos(np.clip((prediction_hwc * target_hwc).sum(axis=2), -1.0, 1.0)))
        rgb = Image.open(str(item["image"])).convert("RGB")
        panels = [
            (f"RGB | {sample_id} | mean {float(sample_metrics['angle_deg']):.2f} deg", np.asarray(rgb)),
            ("GT", eval_normal_to_display_rgb(target_hwc, mask_np)),
            ("Pred", eval_normal_to_display_rgb(prediction_hwc, mask_np)),
            ("Error 0-45deg", error_to_rgb(error_deg, mask_np)),
        ]
        y0 = row_index * (cell_h + label_h)
        for col_index, (title, array) in enumerate(panels):
            panel = Image.fromarray(array).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            x0 = col_index * cell_w
            canvas.paste(panel, (x0, y0 + label_h))
            draw.rectangle((x0, y0, x0 + cell_w, y0 + label_h), fill=(32, 32, 32))
            draw.text((x0 + 8, y0 + 8), title, fill=(235, 235, 235), font=font)

    (method_dir / "representative_rgb_gt_pred_error.png").unlink(missing_ok=True)
    canvas.save(method_dir / "representative_rgb_gt_pred_error.png")
    (method_dir / "representative_rgb_gt_pred_error.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_per_sample_metrics(method_dir: Path) -> dict[str, dict[str, object]]:
    path = method_dir / "per_sample_metrics.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in rows if isinstance(row, dict) and "id" in row}


def representative_sample_ids_for_methods(output_dir: Path, methods: list[str], max_rows: int = 6) -> list[dict[str, object]]:
    by_sample: dict[str, list[float]] = {}
    for method in methods:
        for sample_id, row in load_per_sample_metrics(output_dir / method).items():
            if "angle_deg" not in row:
                continue
            by_sample.setdefault(sample_id, []).append(float(row["angle_deg"]))
    scored = [
        {"id": sample_id, "mean_angle": float(np.mean(angles)), "methods": len(angles)}
        for sample_id, angles in by_sample.items()
        if angles
    ]
    scored.sort(key=lambda row: float(row["mean_angle"]))
    if len(scored) <= max_rows:
        return scored
    quantiles = np.linspace(0.05, 0.98, max_rows)
    selected: list[dict[str, object]] = []
    used: set[int] = set()
    for quantile in quantiles:
        index = int(round(float(quantile) * (len(scored) - 1)))
        while index in used and index + 1 < len(scored):
            index += 1
        used.add(index)
        selected.append(scored[index])
    return selected


def write_method_comparison_visualization(
    *,
    output_dir: Path,
    manifest: list[dict[str, object]],
    methods: list[str],
    dataset: str,
    max_rows: int = 6,
) -> None:
    manifest_by_id = {str(item["id"]): item for item in manifest}
    selected = [
        row for row in representative_sample_ids_for_methods(output_dir, methods, max_rows=max_rows)
        if str(row["id"]) in manifest_by_id
    ]
    if not selected:
        return

    comparison_dir = output_dir / "representative_method_comparison"
    if comparison_dir.exists():
        shutil.rmtree(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "representative_method_comparison.png").unlink(missing_ok=True)

    cell_w, cell_h, label_h = 420, 260, 42
    cols = 3
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    try:
        method_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        metric_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except Exception:
        method_font = font
        metric_font = font

    output_images: list[dict[str, object]] = []
    metric_cache = {method: load_per_sample_metrics(output_dir / method) for method in methods}
    for sample in selected:
        sample_id = str(sample["id"])
        item = manifest_by_id[sample_id]
        target = torch.from_numpy(np.load(str(item["target"])).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        target = convert_target_to_eval(target, dataset)
        target_hwc = target[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        mask_np = np.asarray(Image.open(str(item["mask"])).convert("L")) > 127
        rgb = np.asarray(Image.open(str(item["image"])).convert("RGB"))

        row_count = 1 + len(methods)
        canvas = Image.new("RGB", (cell_w * cols, row_count * (cell_h + label_h)), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)

        def paste_panel(row: int, col: int, title: str, array: np.ndarray, *, fill: tuple[int, int, int] = (32, 32, 32)) -> None:
            x0 = col * cell_w
            y0 = row * (cell_h + label_h)
            draw.rectangle((x0, y0, x0 + cell_w, y0 + label_h), fill=fill)
            draw.text((x0 + 12, y0 + 10), title, fill=(235, 235, 235), font=font)
            panel = Image.fromarray(array).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            canvas.paste(panel, (x0, y0 + label_h))

        blank = np.zeros((target_hwc.shape[0], target_hwc.shape[1], 3), dtype=np.uint8) + 24
        paste_panel(0, 0, f"RGB | {sample_id} | mean {float(sample['mean_angle']):.2f}", rgb)
        paste_panel(0, 1, "GT", eval_normal_to_display_rgb(target_hwc, mask_np))
        paste_panel(0, 2, "Error legend", error_legend_to_rgb(cell_w, cell_h))

        for row_index, method in enumerate(methods, start=1):
            pred_path = find_prediction_path(output_dir / method, sample_id)
            label_array = np.zeros((cell_h, cell_w, 3), dtype=np.uint8) + 32
            label_image = Image.fromarray(label_array)
            label_draw = ImageDraw.Draw(label_image)
            method_metrics = metric_cache[method].get(sample_id, {})
            angle = method_metrics.get("angle_deg")
            metric_text = f"angle {float(angle):.2f} deg" if isinstance(angle, (int, float)) else "missing"
            method_bbox = label_draw.textbbox((0, 0), method, font=method_font)
            metric_bbox = label_draw.textbbox((0, 0), metric_text, font=metric_font)
            method_w = method_bbox[2] - method_bbox[0]
            method_h = method_bbox[3] - method_bbox[1]
            metric_w = metric_bbox[2] - metric_bbox[0]
            metric_h = metric_bbox[3] - metric_bbox[1]
            block_h = method_h + 16 + metric_h
            method_x = max(18, (cell_w - method_w) // 2)
            method_y = max(18, (cell_h - block_h) // 2)
            metric_x = max(18, (cell_w - metric_w) // 2)
            label_draw.text((method_x, method_y), method, fill=(245, 245, 245), font=method_font)
            label_draw.text((metric_x, method_y + method_h + 16), metric_text, fill=(210, 210, 210), font=metric_font)
            if pred_path is None:
                paste_panel(row_index, 0, "Method", np.asarray(label_image))
                paste_panel(row_index, 1, "Pred missing", blank)
                paste_panel(row_index, 2, "Error missing", blank)
                continue
            prediction = convert_prediction_to_eval(load_prediction(pred_path), method)
            if prediction.shape[-2:] != target.shape[-2:]:
                prediction = torch.nn.functional.interpolate(prediction, size=target.shape[-2:], mode="bilinear", align_corners=False)
                prediction = normalize_prediction(prediction)
            pred_hwc = prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
            error_deg = np.degrees(np.arccos(np.clip((pred_hwc * target_hwc).sum(axis=2), -1.0, 1.0)))
            paste_panel(row_index, 0, "Method", np.asarray(label_image))
            paste_panel(row_index, 1, "Pred", eval_normal_to_display_rgb(pred_hwc, mask_np))
            paste_panel(row_index, 2, "Error", error_to_rgb(error_deg, mask_np))

        image_path = comparison_dir / f"{sample_id}_method_comparison.png"
        canvas.save(image_path)
        output_images.append({"id": sample_id, "path": str(image_path), "mean_angle": float(sample["mean_angle"])})

    (output_dir / "representative_method_comparison.json").write_text(
        json.dumps(
            {
                "samples": selected,
                "methods": methods,
                "layout": "one image per representative sample; top row is RGB/GT, then one row per method with Pred/Error.",
                "images": output_images,
                "display_from_eval": {
                    "perm": DISPLAY_FROM_EVAL_PERM,
                    "signs": DISPLAY_FROM_EVAL_SIGNS,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def evaluate_predictions(method_dir: Path, manifest: list[dict[str, object]], method: str, dataset: str = "nyuv2") -> dict[str, float]:
    worker_count = max(1, min(len(manifest), os.cpu_count() or 1))
    if worker_count == 1:
        init_eval_metric_worker(str(method_dir), method, dataset)
        rows = [evaluate_one_sample(item) for item in manifest]
    else:
        with mp.Pool(
            processes=worker_count,
            initializer=init_eval_metric_worker,
            initargs=(str(method_dir), method, dataset),
        ) as pool:
            rows = list(pool.imap(evaluate_one_sample, manifest, chunksize=8))

    missing = [str(row["id"]) for row in rows if row.get("missing")]
    valid_rows = [row for row in rows if not row.get("missing")]
    metric_keys = sorted(key for key in valid_rows[0].keys() if key != "id") if valid_rows else []
    result = {key: float(np.mean([float(row[key]) for row in valid_rows])) for key in metric_keys}
    result.update({"samples": float(len(valid_rows)), "missing": float(len(missing))})
    per_sample: list[dict[str, float | str]] = [
        {
            "id": str(row["id"]),
            "angle_deg": float(row["angle_deg"]),
            "acc_11_25": float(row["acc_11_25"]),
            "acc_22_5": float(row["acc_22_5"]),
            "acc_30": float(row["acc_30"]),
        }
        for row in valid_rows
    ]
    payload = {
        "method": method,
        "official_backend": method != "ours",
        "conventions": convention_payload(dataset, method),
        "metrics": result,
        "missing_ids": missing[:100],
    }
    (method_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (method_dir / "per_sample_metrics.json").write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
    write_representative_visualization(
        method_dir=method_dir,
        manifest=manifest,
        method=method,
        dataset=dataset,
        per_sample=per_sample,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified normal evaluation experiment for toy, NYUv2, Hypersim, and DSINE eval datasets."
    )
    parser.add_argument("--dataset", choices=("toy", "nyuv2", "hypersim", "scannet", "ibims", "sintel"), default="toy")
    parser.add_argument("--data-root", default="auto")
    parser.add_argument("--partition", choices=("train", "val", "test", "ibims", "sintel"), default="val")
    parser.add_argument("--pn", choices=("0.06M", "0.25M", "1M"), default="1M")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--eval-set-workers", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["ours"])
    parser.add_argument("--ours-checkpoint", type=Path, default=ROOT / "outputs/normal_estimation/2026-06-01/09-27-05/checkpoints/best_angle_18.5532.pth")
    parser.add_argument("--normal-tokenizer-ckpt", type=Path, default=ROOT / "outputs/normal_tokenizer/2026-06-03/00-39-35/checkpoints/best_angle_3.5732.pth")
    parser.add_argument("--normal-vae-type", type=int, default=32)
    parser.add_argument("--ours-seed", default="0")
    parser.add_argument("--ours-top-k", default="1")
    parser.add_argument("--ours-top-p", default="0.0")
    parser.add_argument("--ours-tau", default="1.0")
    parser.add_argument("--parallel-shards", default="auto")
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--ours-kv-cache-fast", action="store_true")
    parser.add_argument(
        "--compare-inference-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Measure and compare per-image model inference time only.",
    )
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.ours_checkpoint = resolve_path(args.ours_checkpoint)
    args.normal_tokenizer_ckpt = resolve_path(args.normal_tokenizer_ckpt) if str(args.normal_tokenizer_ckpt) else None
    data_root = dataset_default_root(args.dataset) if args.data_root.lower() == "auto" else resolve_path(args.data_root)

    manifest: list[dict[str, object]] = []
    if args.dataset == "toy":
        input_dir = resolve_image_dir(args.dataset, data_root)
        eval_set_dir = None
    else:
        if args.dataset == "nyuv2" and args.partition == "test":
            raise ValueError("NYUv2 parquet supports train/val, not test.")
        eval_set_dir = output_dir / "_eval_set"
        if args.dry_run:
            input_dir = eval_set_dir / "images"
        else:
            input_dir, manifest = export_eval_set(
                dataset=args.dataset,
                data_root=data_root,
                partition=args.partition,
                pn=args.pn,
                max_samples=args.max_samples,
                work_dir=eval_set_dir,
                workers=args.eval_set_workers,
            )

    methods = parse_methods(args.methods)
    devices = visible_cuda_devices()
    if args.parallel_shards != "auto":
        shard_count = max(1, int(args.parallel_shards))
        devices = devices[:shard_count] if devices else []
    commands: dict[str, object] = {}
    failures: list[str] = []

    meta = {
        "dataset": args.dataset,
        "data_root": str(data_root),
        "partition": args.partition,
        "pn": args.pn,
        "max_samples": args.max_samples,
        "input_dir": str(input_dir),
        "eval_set_dir": str(eval_set_dir) if eval_set_dir else "",
        "methods": methods,
        "ours_checkpoint": str(args.ours_checkpoint),
        "normal_tokenizer_ckpt": str(args.normal_tokenizer_ckpt) if args.normal_tokenizer_ckpt else "",
        "visible_cuda_devices": devices,
        "parallel_shards": args.parallel_shards,
        "compare_inference_time": bool(args.compare_inference_time),
        "timing_warmup": args.timing_warmup,
        "timing_repeats": args.timing_repeats,
        "ours_kv_cache_fast": bool(args.ours_kv_cache_fast),
        "inference_time_protocol": timing_protocol(),
        "normal_conventions": {
            "canonical": EVAL_CONVENTION,
            "dataset_targets": {key: asdict(value) for key, value in DATASET_TARGET_CONVENTIONS.items()},
            "model_outputs": {key: asdict(value) for key, value in MODEL_OUTPUT_CONVENTIONS.items()},
            "display_from_eval": {
                "perm": DISPLAY_FROM_EVAL_PERM,
                "signs": DISPLAY_FROM_EVAL_SIGNS,
                "description": "Normal-map display colors matching infinity.normal_estimation.normals_to_vis: eval (x, y, z) -> RGB (-x, y, z).",
            },
        },
    }
    (output_dir / "eval_experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    timing_by_method: dict[str, dict[str, object]] = {}

    if "ours" in methods:
        ours_out = output_dir / "ours"
        ours_out.mkdir(parents=True, exist_ok=True)
        code, ours_commands = run_ours_sharded(args, input_dir, ours_out, devices)
        timing_by_method["ours"] = load_inference_timing(ours_out, "ours", args.compare_inference_time and not args.dry_run)
        commands["ours"] = ours_commands[0] if len(ours_commands) == 1 else ours_commands
        if code != 0:
            failures.append(f"ours: exited with {code}")
        elif manifest and not args.dry_run:
            evaluate_predictions(ours_out, manifest, "ours", args.dataset)
            update_metrics_timing(ours_out, "ours", timing_by_method["ours"])
        elif not manifest:
            write_image_only_metrics(ours_out, "ours", timing_by_method["ours"])

    baseline_methods = [method for method in methods if method != "ours"]
    if baseline_methods:
        baseline_groups = [[method] for method in baseline_methods] if args.compare_inference_time else [baseline_methods]
        for baseline_group in baseline_groups:
            baseline_cmd = [
                str(PYTHON),
                str(BASELINE_RUNNER),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--methods",
                *baseline_group,
                "--parallel-shards",
                args.parallel_shards,
                "--timing-warmup",
                str(args.timing_warmup),
                "--timing-repeats",
                str(args.timing_repeats),
            ]
            if args.bootstrap:
                baseline_cmd.append("--bootstrap")
            if args.dry_run:
                baseline_cmd.append("--dry-run")
            command_key = baseline_group[0] if len(baseline_group) == 1 else "baselines"
            commands[command_key] = baseline_cmd
            code = run_command(baseline_cmd, cwd=ROOT, dry_run=False, env=no_proxy_env())
            if len(baseline_group) == 1:
                method = baseline_group[0]
                timing_by_method[method] = load_inference_timing(output_dir / method, method, args.compare_inference_time and not args.dry_run)
            if code != 0:
                failures.append(f"{command_key}: exited with {code}")
                continue
            for method in baseline_group:
                if method in timing_by_method:
                    update_metrics_timing(output_dir / method, method, timing_by_method[method])
            if not manifest or args.dry_run:
                continue
            for method in baseline_group:
                try:
                    evaluate_predictions(output_dir / method, manifest, method, args.dataset)
                    if method in timing_by_method:
                        update_metrics_timing(output_dir / method, method, timing_by_method[method])
                except Exception as exc:
                    failures.append(f"{method}: eval failed: {exc}")

    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        metrics_path = output_dir / method / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", payload)
        if isinstance(metrics, dict):
            summary[method] = {str(key): float(value) for key, value in metrics.items() if isinstance(value, (int, float))}
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if timing_by_method:
        def timing_sort_key(item: tuple[str, dict[str, object]]) -> float:
            value = item[1].get("mean_seconds")
            return float(value) if isinstance(value, (int, float)) else float("inf")

        timing_summary = {
            method: timing
            for method, timing in sorted(
                timing_by_method.items(),
                key=timing_sort_key,
            )
        }
        (output_dir / "inference_time_summary.json").write_text(
            json.dumps(
                {
                    "protocol": timing_protocol(),
                    "methods": timing_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if manifest and not args.dry_run:
        write_method_comparison_visualization(
            output_dir=output_dir,
            manifest=manifest,
            methods=[method for method in methods if method in summary],
            dataset=args.dataset,
        )
    (output_dir / "experiment_commands.json").write_text(json.dumps(commands, ensure_ascii=False, indent=2), encoding="utf-8")

    failures_path = output_dir / "experiment_failures.txt"
    if failures:
        failures_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print("\n".join(failures), file=sys.stderr)
        return 2
    if failures_path.exists():
        failures_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
