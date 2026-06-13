#!/usr/bin/env python3
"""Visualize normal differences between DSINE eval data and local training data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from infinity.normal_estimation import NYUv2ParquetNormalDataset
from tools.normal_eval_experiment import convert_target_to_eval, eval_normal_to_display_rgb


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "normal_dataset_preview"
PANEL_W = 224
PANEL_H = 192


def normalize(normal: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / np.maximum(norm, 1e-6)


def read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def read_dsine_png_normal(path: Path, bit_depth: int) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    scale = float((1 << bit_depth) - 1)
    mask = np.sum(rgb, axis=2) > 0
    normal = (rgb.astype(np.float32) / scale) * 2.0 - 1.0
    normal = normalize(normal)
    return normal, mask


def resize_rgb(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def resize_normal(normal: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(normal.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    return normalize(resized)


def resize_mask(mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def angular_error(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dot = np.sum(normalize(a) * normalize(b), axis=-1).clip(-1.0, 1.0)
    err = np.degrees(np.arccos(dot)).astype(np.float32)
    err[~mask] = np.nan
    return err


def error_to_rgb(err: np.ndarray, max_deg: float = 60.0) -> np.ndarray:
    valid = np.isfinite(err)
    scaled = np.zeros_like(err, dtype=np.uint8)
    scaled[valid] = np.clip(err[valid] / max_deg * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    rgb[~valid] = (0, 0, 0)
    return rgb


def rgb_diff_to_rgb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.mean(np.abs(a.astype(np.float32) - b_resized.astype(np.float32)), axis=-1)
    scaled = np.clip(diff / 80.0 * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def rgb_mse(a: np.ndarray, b: np.ndarray) -> float:
    a_thumb = cv2.resize(a, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    b_thumb = cv2.resize(b, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    return float(np.mean((a_thumb - b_thumb) ** 2))


def stats(err: np.ndarray) -> dict[str, float]:
    valid = err[np.isfinite(err)]
    if valid.size == 0:
        return {"mean": math.nan, "median": math.nan, "p95": math.nan}
    return {
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p95": float(np.percentile(valid, 95)),
    }


def panel(image: np.ndarray, label: str) -> Image.Image:
    image = Image.fromarray(image.astype(np.uint8)).resize((PANEL_W, PANEL_H), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (PANEL_W, PANEL_H + 28), (18, 18, 18))
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), label, fill=(235, 235, 235), font=ImageFont.load_default())
    return canvas


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def vkitti2_path_from_dsine_record(record: str) -> tuple[Path, Path, Path]:
    # record: Scene01/clone/frames/rgb/Camera_0/rgb_00032.jpg
    parts = Path(record).parts
    scene, variant, camera, rgb_name = parts[0], parts[1], parts[4], parts[5]
    frame = int(rgb_name.split("_")[-1].split(".")[0])
    rgb_path = ROOT / "data" / "VKITTI2" / "raw" / scene / variant / "frames" / "rgb" / camera / f"rgb_{frame:05d}.jpg"
    processed_root = ROOT / "data" / "VKITTI2" / "processed" / "normals_lotus_svd"
    normal_path = processed_root / "normals" / scene / variant / camera / f"{frame:05d}.npy"
    mask_path = processed_root / "masks" / scene / variant / camera / f"{frame:05d}.png"
    return rgb_path, normal_path, mask_path


def find_vkitti2_match_by_rgb(scene: str, query_rgb: np.ndarray) -> tuple[str, Path, Path, Path, float]:
    query = cv2.resize(query_rgb, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
    best: tuple[float, Path] | None = None
    for rgb_path in sorted((ROOT / "data" / "VKITTI2" / "raw" / scene).glob("*/frames/rgb/*/*.jpg")):
        candidate = read_rgb(rgb_path)
        thumb = cv2.resize(candidate, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        mse = float(np.mean((query - thumb) ** 2))
        if best is None or mse < best[0]:
            best = (mse, rgb_path)
            if mse == 0.0:
                break
    if best is None:
        raise FileNotFoundError(f"No VKITTI2 raw RGB candidates found for {scene}")
    mse, rgb_path = best
    rel = rgb_path.relative_to(ROOT / "data" / "VKITTI2" / "raw").as_posix()
    parts = rgb_path.relative_to(ROOT / "data" / "VKITTI2" / "raw").parts
    frame = int(parts[-1].split("_")[-1].split(".")[0])
    scene_name, variant, camera = parts[0], parts[1], parts[4]
    processed_root = ROOT / "data" / "VKITTI2" / "processed" / "normals_lotus_svd"
    normal_path = processed_root / "normals" / scene_name / variant / camera / f"{frame:05d}.npy"
    mask_path = processed_root / "masks" / scene_name / variant / camera / f"{frame:05d}.png"
    return rel, rgb_path, normal_path, mask_path, mse


def collect_vkitti_rows() -> list[dict[str, Any]]:
    wanted = [("Scene01", 0), ("Scene06", 100)]
    rows: list[dict[str, Any]] = []
    for scene, idx in wanted:
        dsine_root = ROOT / "data" / "vkitti" / scene
        dsine_rgb = dsine_root / f"{idx:06d}_img.jpg"
        dsine_normal_path = dsine_root / f"{idx:06d}_normal.png"
        dsine_rgb_array = read_rgb(dsine_rgb)
        record, vkitti2_rgb_path, vkitti2_normal_path, vkitti2_mask_path, match_mse = find_vkitti2_match_by_rgb(
            scene, dsine_rgb_array
        )
        normal_a, mask_a = read_dsine_png_normal(dsine_normal_path, bit_depth=16)
        normal_b = np.load(vkitti2_normal_path).astype(np.float32)
        mask_b = np.asarray(Image.open(vkitti2_mask_path).convert("L")) > 0
        normal_b = resize_normal(normal_b, normal_a.shape[:2])
        mask_b = resize_mask(mask_b, normal_a.shape[:2])
        mask = mask_a & mask_b
        err = angular_error(normal_a, normal_b, mask)
        rows.append(
            {
                "group": "VKITTI",
                "sample": f"{scene}/{idx:06d}",
                "match": f"{record}, rgb_thumb_mse={match_mse:.1f}",
                "a_rgb": dsine_rgb_array,
                "b_rgb": read_rgb(vkitti2_rgb_path),
                "a_label": "data/vkitti DSINE",
                "a_normal": normal_a,
                "a_mask": mask_a,
                "b_label": "data/VKITTI2 lotus_svd",
                "b_normal": normal_b,
                "b_mask": mask_b,
                "error": err,
                "comparable": True,
                "stats": stats(err),
            }
        )
    return rows


def load_nyuv2_parquet_sample(dataset: NYUv2ParquetNormalDataset, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample = dataset[index]
    image = (sample["image"].permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    target = convert_target_to_eval(sample["target"].unsqueeze(0), "nyuv2")[0].permute(1, 2, 0).numpy()
    mask = sample["mask"].squeeze(0).numpy().astype(bool)
    return image, normalize(target.astype(np.float32)), mask


def find_nyuv2_nearest(dataset: NYUv2ParquetNormalDataset, query_rgb: np.ndarray, limit: int = 654) -> tuple[int, float]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset.files[0], columns=["image"])
    query = cv2.resize(query_rgb, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    best_index = 0
    best_mse = float("inf")
    for index in range(min(limit, len(dataset))):
        file_index, row_index = dataset.records[index]
        if file_index != 0:
            break
        image = np.asarray(table["image"][row_index].as_py(), dtype=np.float32)
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        image = (image.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        thumb = cv2.resize(image, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        mse = float(np.mean((query - thumb) ** 2))
        if mse < best_mse:
            best_index = index
            best_mse = mse
    return best_index, best_mse


def collect_nyuv2_rows() -> list[dict[str, Any]]:
    dataset = NYUv2ParquetNormalDataset(
        root=str(ROOT / "data" / "NYUv2" / "hf-parquet" / "tanganke" / "nyuv2" / "data"),
        partition="val",
        pn="1M",
    )
    rows: list[dict[str, Any]] = []
    dsine_images = sorted((ROOT / "data" / "nyuv2" / "test").glob("*_img.png"))
    chosen = [0, len(dsine_images) // 3, len(dsine_images) - 1]
    for ordinal in chosen:
        dsine_rgb_path = dsine_images[ordinal]
        idx = int(dsine_rgb_path.name.split("_")[0])
        dsine_normal_path = dsine_rgb_path.with_name(dsine_rgb_path.name.replace("_img.png", "_normal.png"))
        dsine_rgb = read_rgb(dsine_rgb_path)
        normal_a, mask_a = read_dsine_png_normal(dsine_normal_path, bit_depth=8)
        parquet_rank = min(ordinal, len(dataset) - 1)
        parquet_rgb, normal_b, mask_b = load_nyuv2_parquet_sample(dataset, parquet_rank)
        mse = rgb_mse(dsine_rgb, parquet_rgb)
        normal_b = resize_normal(normal_b, normal_a.shape[:2])
        mask_b = resize_mask(mask_b, normal_a.shape[:2])
        mask = mask_a & mask_b
        comparable = mse < 25.0
        err = angular_error(normal_a, normal_b, mask) if comparable else np.full(normal_a.shape[:2], np.nan, dtype=np.float32)
        rows.append(
            {
                "group": "NYUv2",
                "sample": f"test/{idx:06d} (rank {ordinal})",
                "match": f"val/{parquet_rank:06d}, rgb_thumb_mse={mse:.1f}",
                "a_rgb": dsine_rgb,
                "b_rgb": parquet_rgb,
                "a_label": "data/nyuv2 GeoNet",
                "a_normal": normal_a,
                "a_mask": mask_a,
                "b_label": "data/NYUv2 parquet",
                "b_normal": normal_b,
                "b_mask": mask_b,
                "error": err,
                "comparable": comparable,
                "stats": stats(err),
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row_images: list[Image.Image] = []
    summary = []
    for row in rows:
        group = row["group"]
        sample = row["sample"].replace("/", "_")
        prefix = f"{group.lower()}_{sample}"
        normal_a_vis = eval_normal_to_display_rgb(row["a_normal"], row["a_mask"])
        normal_b_vis = eval_normal_to_display_rgb(row["b_normal"], row["b_mask"])
        error_vis = error_to_rgb(row["error"]) if row["comparable"] else rgb_diff_to_rgb(row["a_rgb"], row["b_rgb"])
        Image.fromarray(error_vis).save(OUT_DIR / f"{prefix}_error.png")
        stat = row["stats"]
        title = f"{group} {row['sample']} -> {row['match']}"
        rgb_error = rgb_mse(row["a_rgb"], row["b_rgb"])
        if row["comparable"]:
            err_label = f"normal error mean {stat['mean']:.1f} med {stat['median']:.1f} p95 {stat['p95']:.1f}"
        else:
            err_label = f"RGB differs, no normal error; mse {rgb_error:.1f}"
        panels = [
            panel(row["a_rgb"], title[:52]),
            panel(row["b_rgb"], f"matched RGB mse {rgb_error:.1f}"),
            panel(normal_a_vis, row["a_label"]),
            panel(normal_b_vis, row["b_label"]),
            panel(error_vis, err_label),
        ]
        canvas = Image.new("RGB", (PANEL_W * len(panels), PANEL_H + 28), (18, 18, 18))
        for col, item in enumerate(panels):
            canvas.paste(item, (col * PANEL_W, 0))
        row_images.append(canvas)
        summary.append(
            {
                "group": row["group"],
                "sample": row["sample"],
                "match": row["match"],
                "a_label": row["a_label"],
                "b_label": row["b_label"],
                "rgb_thumb_mse": rgb_error,
                "same_rgb_comparable": bool(row["comparable"]),
                "angular_error_deg": row["stats"],
            }
        )
    final = Image.new("RGB", (PANEL_W * 5, (PANEL_H + 28) * len(row_images)), (12, 12, 12))
    for row_idx, image in enumerate(row_images):
        final.paste(image, (0, row_idx * (PANEL_H + 28)))
    final.save(OUT_DIR / "nyuv2_vkitti_normal_source_comparison.png")
    (OUT_DIR / "nyuv2_vkitti_normal_source_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = collect_vkitti_rows() + collect_nyuv2_rows()
    write_outputs(rows)
    print(OUT_DIR / "nyuv2_vkitti_normal_source_comparison.png")


if __name__ == "__main__":
    main()
