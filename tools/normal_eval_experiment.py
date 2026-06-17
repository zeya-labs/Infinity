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

from infinity.normal_estimation.defaults import DEFAULT_NORMAL_ESTIMATION_CKPT, DEFAULT_NORMAL_TOKENIZER_CKPT  # noqa: E402

PYTHON = ROOT / ".venv" / "bin" / "python"
BASELINE_RUNNER = ROOT / "external" / "normal_baselines" / "scripts" / "run_normal_baseline_compare.py"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_EXPORT_DATASET = None
_EXPORT_DATASET_NAME: str | None = None
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


EVAL_CONVENTION = "camera_left_up_backward"
IDENTITY_CONVENTION = NormalConvention(
    name=EVAL_CONVENTION,
    to_eval_perm=(0, 1, 2),
    to_eval_signs=(1, 1, 1),
    description="Canonical metric convention: +x left, +y up, +z backward.",
)
NYUV2_RIGHT_FORWARD_UP_CONVENTION = NormalConvention(
    name="nyuv2_raw_right_forward_up",
    to_eval_perm=(0, 2, 1),
    to_eval_signs=(-1, 1, -1),
    description="NYUv2 parquet normals. Canonical mapping: (x, y, z) -> (-x, z, -y).",
)
GT_EXPORT_DATASETS = ("nyuv2", "hypersim", "scannet", "ibims", "sintel")

# Dataset readers return raw target tensors. Export normalizes each dataset into
# EVAL_CONVENTION before writing _eval_set/gt/*.npy. Keep this table explicit so
# adding a dataset means declaring its raw target convention in one place.
DATASET_TARGET_CONVENTIONS: dict[str, NormalConvention] = {
    "nyuv2": NYUV2_RIGHT_FORWARD_UP_CONVENTION,
    "hypersim": IDENTITY_CONVENTION,
    "scannet": IDENTITY_CONVENTION,
    "ibims": IDENTITY_CONVENTION,
    "sintel": IDENTITY_CONVENTION,
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
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(-1, 1, 1),
        description="Marigold normals_npy output. Canonical mapping: (x, y, z) -> (-x, y, z).",
    ),
    "geowizard": NormalConvention(
        name="geowizard_normal_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="GeoWizard normal_npy output already matches the canonical convention.",
    ),
    "stablenormal": NormalConvention(
        name="stablenormal_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="StableNormal output already matches the canonical convention.",
    ),
    "lotusg": NormalConvention(
        name="lotus_g_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Lotus-G output already matches the canonical convention.",
    ),
    "lotusd": NormalConvention(
        name="lotus_d_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Lotus-D output already matches the canonical convention.",
    ),
    "dsine": NormalConvention(
        name="dsine_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="DSINE output already matches the canonical convention.",
    ),
    "metric3dv2": NormalConvention(
        name="metric3d_v2_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(-1, -1, -1),
        description="Metric3D v2 output. Canonical mapping: (x, y, z) -> (-x, -y, -z).",
    ),
    "omnidata_v2": NormalConvention(
        name="omnidata_v2_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Omnidata V2 adapter output already matches the canonical convention.",
    ),
    "marigold_e2eft": NormalConvention(
        name="marigold_e2eft_xyz",
        to_eval_perm=(0, 1, 2),
        to_eval_signs=(1, 1, 1),
        description="Marigold E2E-FT output already matches the canonical convention.",
    ),
}

# Direct normal-map visualization in the canonical convention.
DISPLAY_FROM_EVAL_PERM = (0, 1, 2)
DISPLAY_FROM_EVAL_SIGNS = (1, 1, 1)


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
        return ROOT / "data"
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
    global _EXPORT_DATASET, _EXPORT_DATASET_NAME, _EXPORT_WORK_DIR
    torch.set_num_threads(1)
    from infinity.normal_estimation import DSINEEvalNormalDataset, HypersimNormalDataset, NYUv2ParquetNormalDataset

    _EXPORT_DATASET_NAME = dataset
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
    if _EXPORT_DATASET is None or _EXPORT_DATASET_NAME is None or _EXPORT_WORK_DIR is None:
        raise RuntimeError("Export worker is not initialized.")
    sample = _EXPORT_DATASET[index]
    metadata = dict(sample["metadata"])
    sample_id = sample_id_from_metadata(index, metadata)
    image_path = _EXPORT_WORK_DIR / "images" / f"{sample_id}.png"
    target_path = _EXPORT_WORK_DIR / "gt" / f"{sample_id}_normal.npy"
    target_vis_path = target_path.with_suffix(".png")
    mask_path = _EXPORT_WORK_DIR / "mask" / f"{sample_id}_mask.png"
    save_image_tensor(sample["image"], image_path)
    target_eval = convert_target_to_eval(sample["target"].unsqueeze(0), _EXPORT_DATASET_NAME)[0]
    target_hwc = target_eval.permute(1, 2, 0).cpu().numpy().astype(np.float32)
    mask_np = sample["mask"].squeeze(0).cpu().numpy().astype(bool)
    np.save(target_path, target_hwc)
    Image.fromarray(eval_normal_to_display_rgb(target_hwc, mask_np)).save(target_vis_path)
    Image.fromarray(mask_np.astype(np.uint8) * 255).save(mask_path)
    return {
        "id": sample_id,
        "image": str(image_path),
        "target": str(target_path),
        "target_visualization": str(target_vis_path),
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


def infer_eval_set_dir_from_image(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent
    raise ValueError(f"--eval-image must point to an image under an _eval_set/images directory: {image_path}")


def copy_or_link_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def materialize_single_eval_image(eval_image: Path, output_eval_set_dir: Path) -> tuple[Path, list[dict[str, object]], Path]:
    image_path = resolve_path(eval_image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"--eval-image not found: {image_path}")
    source_eval_set_dir = infer_eval_set_dir_from_image(image_path)
    source_manifest_path = source_eval_set_dir / "manifest.jsonl"
    if output_eval_set_dir.exists():
        shutil.rmtree(output_eval_set_dir)
    for name in ("images", "gt", "mask"):
        (output_eval_set_dir / name).mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    manifest_item: dict[str, object] | None = None
    if source_manifest_path.is_file():
        for item in load_eval_manifest(source_eval_set_dir):
            if Path(str(item.get("image", ""))).resolve() == image_path or str(item.get("id", "")) == stem:
                manifest_item = dict(item)
                break
    if manifest_item is None:
        manifest_item = {
            "id": stem,
            "image": str(image_path),
            "target": str(source_eval_set_dir / "gt" / f"{stem}_normal.npy"),
            "target_visualization": str(source_eval_set_dir / "gt" / f"{stem}_normal.png"),
            "mask": str(source_eval_set_dir / "mask" / f"{stem}_mask.png"),
            "source_image": "",
            "target_size": [],
        }

    required = {
        "image": Path(str(manifest_item["image"])),
        "target": Path(str(manifest_item["target"])),
        "mask": Path(str(manifest_item["mask"])),
    }
    missing = [f"{key}={path}" for key, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Single-image eval set is incomplete: " + ", ".join(missing))

    target_vis = Path(str(manifest_item.get("target_visualization", "")))
    copied_item = dict(manifest_item)
    copied_item["image"] = str(output_eval_set_dir / "images" / image_path.name)
    copied_item["target"] = str(output_eval_set_dir / "gt" / required["target"].name)
    copied_item["mask"] = str(output_eval_set_dir / "mask" / required["mask"].name)
    if target_vis.is_file():
        copied_item["target_visualization"] = str(output_eval_set_dir / "gt" / target_vis.name)

    copy_or_link_file(image_path, Path(str(copied_item["image"])))
    copy_or_link_file(required["target"], Path(str(copied_item["target"])))
    copy_or_link_file(required["mask"], Path(str(copied_item["mask"])))
    if target_vis.is_file():
        copy_or_link_file(target_vis, Path(str(copied_item["target_visualization"])))
    (output_eval_set_dir / "manifest.jsonl").write_text(
        json.dumps(copied_item, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_eval_set_dir / "images", [copied_item], source_eval_set_dir


def infer_dataset_from_eval_set(eval_set_dir: Path) -> str | None:
    meta_path = eval_set_dir.parent / "eval_experiment.json"
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    dataset = payload.get("dataset")
    return str(dataset) if isinstance(dataset, str) and dataset else None


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env: dict[str, str],
    cuda_device: str | None = None,
    log_path: Path | None = None,
) -> int:
    prefix = f"CUDA_VISIBLE_DEVICES={cuda_device} " if cuda_device is not None else ""
    print("$ " + prefix + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return 0
    if log_path is None:
        return int(subprocess.run([str(part) for part in cmd], cwd=str(cwd), env=env, check=False).returncode)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + prefix + " ".join(str(part) for part in cmd) + "\n")
        log_file.flush()
        return int(
            subprocess.run(
                [str(part) for part in cmd],
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        )


def ours_command(args: argparse.Namespace, input_dir: Path, output_dir: Path, checkpoint: Path) -> list[str]:
    cmd = [
        str(PYTHON),
        "tools/run_normal_estimation.py",
        "--model-path",
        str(checkpoint),
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
    if os.environ.get("INFINITY_NORMAL_DISABLE_KV_FAST", "").lower() in {"1", "yes", "true", "y"}:
        cmd.append("--normal-disable-kv-cache-fast")
    if os.environ.get("INFINITY_NORMAL_FORCE_ORIGINAL_RESOLUTION", "").lower() in {"1", "yes", "true", "y"}:
        cmd.append("--force-original-resolution")
    return cmd


def run_ours_sharded(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    devices: list[str],
    checkpoint: Path,
) -> tuple[int, list[list[str]]]:
    shard_count = max(1, len(devices))
    input_shards = prepare_input_shards(input_dir, output_dir / "_input_shards", shard_count)
    if len(input_shards) == 1:
        cmd = ours_command(args, input_shards[0], output_dir, checkpoint)
        device = devices[0] if devices else None
        code = run_command(
            cmd,
            cwd=ROOT,
            dry_run=args.dry_run,
            env=clean_env(device),
            cuda_device=device,
            log_path=output_dir / "command.log",
        )
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
        cmd = ours_command(args, shard_input, shard_output, checkpoint)
        commands.append(cmd)
        device = devices[index % len(devices)] if devices else None
        print(f"$ CUDA_VISIBLE_DEVICES={device} " + " ".join(cmd), flush=True)
        if not args.dry_run:
            log_path = shard_output / "command.log"
            log_file = log_path.open("w", encoding="utf-8")
            log_file.write(f"$ CUDA_VISIBLE_DEVICES={device} " + " ".join(cmd) + "\n")
            log_file.flush()
            processes.append(
                subprocess.Popen(cmd, cwd=str(ROOT), env=clean_env(device), stdout=log_file, stderr=subprocess.STDOUT)
            )
    codes = [process.wait() for process in processes] if not args.dry_run else [0 for _ in commands]
    if any(code != 0 for code in codes):
        return 2, commands
    if not args.dry_run:
        merge_shard_outputs([shard_output_root / f"shard_{index:02d}" for index in range(len(input_shards))], output_dir)
    return 0, commands


def run_ours_method(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    devices: list[str],
    checkpoint: Path,
) -> tuple[int, list[list[str]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return run_ours_sharded(args, input_dir, output_dir, devices, checkpoint)


def finalize_ours_method(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    method: str,
    manifest: list[dict[str, object]],
    dataset: str,
) -> dict[str, object]:
    timing = load_inference_timing(output_dir, method, args.compare_inference_time and not args.dry_run)
    if manifest and not args.dry_run:
        evaluate_predictions(output_dir, manifest, method, dataset)
        update_metrics_timing(output_dir, method, timing)
        write_canonical_predictions(output_dir, sample_ids_for_outputs(input_dir, manifest), method, manifest)
    elif not manifest:
        write_image_only_metrics(output_dir, method, timing)
        if not args.dry_run:
            write_canonical_predictions(output_dir, sample_ids_for_outputs(input_dir, manifest), method, manifest)
    return timing


def run_ours_methods(
    *,
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    methods: list[str],
    checkpoints: dict[str, Path],
    devices: list[str],
    manifest: list[dict[str, object]],
    dataset: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]], list[str]]:
    commands: dict[str, object] = {}
    timing_by_method: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    if len(methods) <= 1 or len(devices) <= 1:
        for method in methods:
            method_out = output_dir / method
            code, method_commands = run_ours_method(args, input_dir, method_out, devices, checkpoints[method])
            commands[method] = method_commands[0] if len(method_commands) == 1 else method_commands
            if code != 0:
                failures.append(f"{method}: exited with {code}")
                continue
            timing_by_method[method] = finalize_ours_method(
                args=args,
                input_dir=input_dir,
                output_dir=method_out,
                method=method,
                manifest=manifest,
                dataset=dataset,
            )
        return commands, timing_by_method, failures

    running: list[tuple[str, Path, str, list[str], subprocess.Popen[bytes]]] = []
    pending = list(methods)
    active_limit = min(len(devices), len(pending))
    available_devices = list(devices)
    while pending or running:
        while pending and len(running) < active_limit and available_devices:
            method = pending.pop(0)
            method_out = output_dir / method
            method_out.mkdir(parents=True, exist_ok=True)
            device = available_devices.pop(0)
            cmd = ours_command(args, input_dir, method_out, checkpoints[method])
            commands[method] = cmd
            print(f"$ CUDA_VISIBLE_DEVICES={device} " + " ".join(cmd), flush=True)
            if args.dry_run:
                timing_by_method[method] = load_inference_timing(method_out, method, False)
                available_devices.append(device)
                continue
            running.append((method, method_out, device, cmd, subprocess.Popen(cmd, cwd=str(ROOT), env=clean_env(device))))
        if args.dry_run:
            continue
        method, method_out, device, _cmd, process = running.pop(0)
        code = process.wait()
        available_devices.append(device)
        if code != 0:
            failures.append(f"{method}: exited with {code}")
            continue
        timing_by_method[method] = finalize_ours_method(
            args=args,
            input_dir=input_dir,
            output_dir=method_out,
            method=method,
            manifest=manifest,
            dataset=dataset,
        )
    return commands, timing_by_method, failures


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


def method_convention_key(method: str) -> str:
    return "ours" if method.startswith("ours__") else method


def model_output_convention(method: str) -> NormalConvention:
    key = method_convention_key(method)
    if key not in MODEL_OUTPUT_CONVENTIONS:
        raise ValueError(f"No output normal convention registered for method={method!r}")
    return MODEL_OUTPUT_CONVENTIONS[key]


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
            "description": "Direct normal-map display: RGB = (canonical normal + 1) / 2.",
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
        if not any(part.startswith("_") or part == "canonical_predictions" for part in path.relative_to(method_dir).parts)
    )
    return matches[0] if matches else None


def sample_ids_for_outputs(input_dir: Path, manifest: list[dict[str, object]]) -> list[str]:
    if manifest:
        return [str(item["id"]) for item in manifest]
    return [path.stem for path in resolve_image_paths(input_dir)]


def _resize_mask(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    if mask.shape == size_hw:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def _canonical_metrics_payload(canonical_dir: Path, count: int) -> dict[str, object]:
    return {
        "dir": str(canonical_dir),
        "num_png": count,
        "num_npy": count,
        "convention": EVAL_CONVENTION,
        "display": {
            "perm": DISPLAY_FROM_EVAL_PERM,
            "signs": DISPLAY_FROM_EVAL_SIGNS,
            "description": "Direct normal-map display: RGB = (canonical normal + 1) / 2.",
        },
    }


def attach_canonical_metrics(method_dir: Path, canonical_dir: Path, count: int) -> None:
    path = method_dir / "metrics.json"
    payload: dict[str, object]
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        payload = loaded if isinstance(loaded, dict) else {}
    else:
        payload = {}
    payload["canonical_predictions"] = _canonical_metrics_payload(canonical_dir, count)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_canonical_predictions(
    method_dir: Path,
    sample_ids: list[str],
    method: str,
    manifest: list[dict[str, object]] | None = None,
) -> int:
    canonical_dir = method_dir / "canonical_predictions"
    if canonical_dir.exists():
        shutil.rmtree(canonical_dir)
    canonical_dir.mkdir(parents=True, exist_ok=True)
    manifest_by_id = {str(item["id"]): item for item in manifest or []}
    count = 0
    for sample_id in sample_ids:
        pred_path = find_prediction_path(method_dir, sample_id)
        if pred_path is None:
            continue
        prediction = convert_prediction_to_eval(load_prediction(pred_path), method)
        prediction_hwc = prediction[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        mask_np = None
        item = manifest_by_id.get(sample_id)
        if item is not None and item.get("mask"):
            mask_np = np.asarray(Image.open(str(item["mask"])).convert("L")) > 127
            mask_np = _resize_mask(mask_np, prediction_hwc.shape[:2])
        np.save(canonical_dir / f"{sample_id}_normal.npy", prediction_hwc)
        Image.fromarray(eval_normal_to_display_rgb(prediction_hwc, mask_np)).save(canonical_dir / f"{sample_id}_normal.png")
        count += 1
    attach_canonical_metrics(method_dir, canonical_dir, count)
    return count


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
    target = normalize_prediction(target)
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
        target = normalize_prediction(target)
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
        target = normalize_prediction(target)
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
    parser.add_argument("--eval-image", type=Path, default=None, help="Evaluate one image from an existing _eval_set/images dir.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["ours"])
    parser.add_argument("--ours-checkpoint", type=Path, nargs="+", default=[])
    parser.add_argument("--normal-tokenizer-ckpt", type=Path, default=Path(DEFAULT_NORMAL_TOKENIZER_CKPT))
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
    args.ours_checkpoint = [resolve_path(path) for path in args.ours_checkpoint]
    args.normal_tokenizer_ckpt = resolve_path(args.normal_tokenizer_ckpt) if str(args.normal_tokenizer_ckpt) else None
    data_root = dataset_default_root(args.dataset) if args.data_root.lower() == "auto" else resolve_path(args.data_root)

    manifest: list[dict[str, object]] = []
    source_eval_set_dir = None
    if args.eval_image is not None:
        eval_set_dir = output_dir / "_eval_set"
        input_dir, manifest, source_eval_set_dir = materialize_single_eval_image(args.eval_image, eval_set_dir)
        inferred_dataset = infer_dataset_from_eval_set(source_eval_set_dir)
        if inferred_dataset:
            args.dataset = inferred_dataset
            if args.data_root.lower() == "auto":
                data_root = dataset_default_root(args.dataset)
    elif args.dataset == "toy":
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
    ours_checkpoints = list(args.ours_checkpoint)
    has_ours = any(method == "ours" or method.startswith("ours__") for method in methods)
    if has_ours and not ours_checkpoints:
        ours_checkpoints = [resolve_path(Path(DEFAULT_NORMAL_ESTIMATION_CKPT))]
    if "ours" in methods and len(ours_checkpoints) > 1:
        expanded_methods: list[str] = []
        for method in methods:
            if method != "ours":
                expanded_methods.append(method)
                continue
            for checkpoint in ours_checkpoints:
                expanded_methods.append("ours__" + sanitize_sample_id(checkpoint.stem))
        methods = expanded_methods
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
        "source_eval_set_dir": str(source_eval_set_dir) if source_eval_set_dir else "",
        "eval_image": str(resolve_path(args.eval_image)) if args.eval_image else "",
        "methods": methods,
        "ours_checkpoints": [str(path) for path in ours_checkpoints],
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
                "description": "Direct normal-map display: RGB = (canonical normal + 1) / 2.",
            },
        },
    }
    (output_dir / "eval_experiment.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)

    timing_by_method: dict[str, dict[str, object]] = {}

    ours_methods = [method for method in methods if method == "ours" or method.startswith("ours__")]
    if not ours_methods:
        ours_checkpoint_by_method = {}
    elif ours_methods == ["ours"]:
        ours_checkpoint_by_method = {"ours": ours_checkpoints[0]}
    else:
        ours_checkpoint_by_method = {
            method: checkpoint for method, checkpoint in zip(ours_methods, ours_checkpoints, strict=True)
        }
    ours_commands, ours_timing, ours_failures = run_ours_methods(
        args=args,
        input_dir=input_dir,
        output_dir=output_dir,
        methods=ours_methods,
        checkpoints=ours_checkpoint_by_method,
        devices=devices,
        manifest=manifest,
        dataset=args.dataset,
    )
    commands.update(ours_commands)
    timing_by_method.update(ours_timing)
    failures.extend(ours_failures)

    baseline_methods = [method for method in methods if method not in ours_methods]
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
                if not args.dry_run:
                    sample_ids = sample_ids_for_outputs(input_dir, manifest)
                    for method in baseline_group:
                        write_canonical_predictions(output_dir / method, sample_ids, method, manifest)
                continue
            for method in baseline_group:
                try:
                    evaluate_predictions(output_dir / method, manifest, method, args.dataset)
                    if method in timing_by_method:
                        update_metrics_timing(output_dir / method, method, timing_by_method[method])
                    write_canonical_predictions(
                        output_dir / method,
                        sample_ids_for_outputs(input_dir, manifest),
                        method,
                        manifest,
                    )
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
