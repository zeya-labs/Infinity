#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate surface normals from VKITTI2 depth maps.")
    parser.add_argument("--root", type=Path, default=ROOT_DIR / "data/VKITTI2/raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "data/VKITTI2/processed/normals_from_depth")
    parser.add_argument("--depth-scale", type=float, default=100.0, help="Depth PNG scale. VKITTI2 depth is commonly stored in centimeters.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-vis", action="store_true", default=False)
    return parser.parse_args()


def find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match under {root}: {pattern}")
    return matches[0]


def load_depth(path: Path, scale: float) -> np.ndarray:
    depth = np.asarray(Image.open(path), dtype=np.float32)
    depth = depth / float(scale)
    depth[~np.isfinite(depth)] = 0
    return depth


def frame_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(match.group(1))


def camera_name(path: Path) -> str:
    for part in path.parts:
        if part.lower().startswith("camera_"):
            return part
    return "Camera_0"


def scene_variant(path: Path) -> tuple[str, str]:
    parts = path.parts
    for i, part in enumerate(parts):
        if part.startswith("Scene") and i + 1 < len(parts):
            return part, parts[i + 1]
    return "unknown_scene", "unknown_variant"


def load_intrinsics(textgt_root: Path) -> dict[tuple[str, str, str, int], tuple[float, float, float, float]]:
    intrinsics: dict[tuple[str, str, str, int], tuple[float, float, float, float]] = {}
    for path in sorted(textgt_root.glob("Scene*/**/intrinsic.txt")):
        scene, variant = scene_variant(path)
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=" ")
            for row in reader:
                row = {k: v for k, v in row.items() if k is not None and k != ""}
                if not row:
                    continue
                frame = int(row.get("frame", row.get("frame_id", row.get("Frame", 0))))
                camera = row.get("cameraID", row.get("camera", row.get("Camera", "0")))
                camera = f"Camera_{camera}" if str(camera).isdigit() else str(camera)
                fx = pick_float(row, ("K[0,0]", "fx", "focal_x"))
                fy = pick_float(row, ("K[1,1]", "fy", "focal_y"))
                cx = pick_float(row, ("K[0,2]", "cx", "center_x"))
                cy = pick_float(row, ("K[1,2]", "cy", "center_y"))
                intrinsics[(scene, variant, camera, frame)] = (fx, fy, cx, cy)
    return intrinsics


def pick_float(row: dict[str, str], names: tuple[str, ...]) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(f"Missing any of {names} in intrinsic row keys={sorted(row)}")


def default_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    # KITTI-like fallback. Use only if textgt intrinsic.txt is unavailable.
    return 725.0, 725.0, (width - 1) / 2.0, (height - 1) / 2.0


def depth_to_normals(depth: np.ndarray, intrinsics: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    z = depth
    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)

    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dx[:, 0] = points[:, 1] - points[:, 0]
    dx[:, -1] = points[:, -1] - points[:, -2]
    dy[1:-1] = points[2:] - points[:-2]
    dy[0] = points[1] - points[0]
    dy[-1] = points[-1] - points[-2]

    normal = np.cross(dx, dy)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    valid = (depth > 0) & np.isfinite(depth) & (norm[..., 0] > 1e-6)
    normal = normal / np.maximum(norm, 1e-6)
    # Orient normals toward the camera. Camera looks along +z in this convention.
    flip = normal[..., 2] > 0
    normal[flip] *= -1
    normal[~valid] = 0
    return normal.astype(np.float32), valid


def normal_vis(normal: np.ndarray) -> Image.Image:
    vis = ((normal.clip(-1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(vis, mode="RGB")


def corresponding_rgb(depth_path: Path, rgb_root: Path) -> Path | None:
    scene, variant = scene_variant(depth_path)
    camera = camera_name(depth_path)
    frame = frame_index(depth_path)
    candidates = [
        rgb_root / scene / variant / "frames" / "rgb" / camera / f"rgb_{frame:05d}.jpg",
        rgb_root / scene / variant / "frames" / "rgb" / camera / f"rgb_{frame:05d}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    args = parse_args()
    depth_root = find_one(args.root, "vkitti_2.0.3_depth")
    rgb_root = find_one(args.root, "vkitti_2.0.3_rgb")
    textgt_root = find_one(args.root, "vkitti_2.0.3_textgt")

    intrinsics = load_intrinsics(textgt_root)
    depth_files = sorted(depth_root.glob("Scene*/**/frames/depth/Camera_*/*.png"))
    if args.max_samples > 0:
        depth_files = depth_files[: args.max_samples]
    if not depth_files:
        raise FileNotFoundError(f"No VKITTI2 depth PNG files found under {depth_root}")

    normal_dir = args.out_dir / "normals"
    mask_dir = args.out_dir / "masks"
    vis_dir = args.out_dir / "normal_vis"
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for idx, depth_path in enumerate(depth_files):
        scene, variant = scene_variant(depth_path)
        camera = camera_name(depth_path)
        frame = frame_index(depth_path)
        depth = load_depth(depth_path, args.depth_scale)
        k = intrinsics.get((scene, variant, camera, frame), default_intrinsics(depth.shape[1], depth.shape[0]))
        normal, valid = depth_to_normals(depth, k)

        rel = Path(scene) / variant / camera / f"{frame:05d}"
        normal_path = normal_dir / rel.with_suffix(".npy")
        mask_path = mask_dir / rel.with_suffix(".png")
        normal_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(normal_path, normal)
        Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(mask_path)
        vis_path = None
        if args.save_vis:
            vis_path = vis_dir / rel.with_suffix(".png")
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            normal_vis(normal).save(vis_path)

        rgb_path = corresponding_rgb(depth_path, rgb_root)
        rows.append(
            {
                "scene": scene,
                "variant": variant,
                "camera": camera,
                "frame": frame,
                "rgb_path": str(rgb_path) if rgb_path else "",
                "depth_path": str(depth_path),
                "normal_path": str(normal_path),
                "mask_path": str(mask_path),
                "normal_vis_path": str(vis_path) if vis_path else "",
                "fx": k[0],
                "fy": k[1],
                "cx": k[2],
                "cy": k[3],
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1}/{len(depth_files)}", flush=True)

    manifest = args.out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
