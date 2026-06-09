#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate surface normals from VKITTI2 depth maps.")
    parser.add_argument("--root", type=Path, default=ROOT_DIR / "data/VKITTI2/raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "data/VKITTI2/processed/normals_from_depth")
    parser.add_argument("--depth-scale", type=float, default=100.0, help="Depth PNG scale. VKITTI2 depth is commonly stored in centimeters.")
    parser.add_argument("--method", choices=("d2nt_basic", "d2nt_v2", "d2nt_v3", "lotus_svd"), default="d2nt_v3")
    parser.add_argument("--no-flip", action="store_true", help="Do not flip normals to VKITTI2 camera-forward convention.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for lotus_svd generation.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for lotus_svd generation.")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--save-vis", action="store_true", default=False)
    return parser.parse_args()


def resolve_vkitti2_root(root: Path, dataset_kind: str) -> Path:
    candidates = [root / f"vkitti_2.0.3_{dataset_kind}", root]
    for candidate in candidates:
        if dataset_kind in {"rgb", "depth"}:
            pattern = f"Scene*/**/frames/{dataset_kind}/Camera_*"
        else:
            pattern = "Scene*/**/intrinsic.txt"
        if candidate.is_dir() and any(candidate.glob(pattern)):
            return candidate
    raise FileNotFoundError(f"No VKITTI2 {dataset_kind} tree found under {root}")


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


KERNEL_GX = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=np.float32)
KERNEL_GY = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.float32)
GRADIENT_L = np.array([[-1, 1, 0]], dtype=np.float32)
GRADIENT_R = np.array([[0, -1, 1]], dtype=np.float32)
GRADIENT_U = np.array([[-1], [1], [0]], dtype=np.float32)
GRADIENT_D = np.array([[0], [-1], [1]], dtype=np.float32)
LAP_KER_ALPHA = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)


def vector_normalization(normal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mag = np.linalg.norm(normal, axis=2, keepdims=True) + eps
    return (normal / mag).astype(np.float32)


def soft_min(laplace_map: np.ndarray, base: float, direction: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = laplace_map.shape
    eps = 1e-8
    lap_power = np.power(base, -laplace_map).astype(np.float32)
    if direction == 0:
        lap_pow_l = np.hstack([np.zeros((height, 1), dtype=np.float32), lap_power[:, :-1]])
        lap_pow_r = np.hstack([lap_power[:, 1:], np.zeros((height, 1), dtype=np.float32)])
        return (
            (lap_pow_l + eps * 0.5) / (eps + lap_pow_l + lap_pow_r),
            (lap_pow_r + eps * 0.5) / (eps + lap_pow_l + lap_pow_r),
        )
    lap_pow_u = np.vstack([np.zeros((1, width), dtype=np.float32), lap_power[:-1, :]])
    lap_pow_d = np.vstack([lap_power[1:, :], np.zeros((1, width), dtype=np.float32)])
    return (
        (lap_pow_u + eps * 0.5) / (eps + lap_pow_u + lap_pow_d),
        (lap_pow_d + eps * 0.5) / (eps + lap_pow_u + lap_pow_d),
    )


def get_filter(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    gu = cv2.filter2D(depth, -1, KERNEL_GX) / 2
    gv = cv2.filter2D(depth, -1, KERNEL_GY) / 2
    return gu.astype(np.float32), gv.astype(np.float32)


def get_dag_filter(depth: np.ndarray, base: float = np.e) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    grad_l = cv2.filter2D(depth, -1, GRADIENT_L).astype(np.float32)
    grad_r = cv2.filter2D(depth, -1, GRADIENT_R).astype(np.float32)
    grad_u = cv2.filter2D(depth, -1, GRADIENT_U).astype(np.float32)
    grad_d = cv2.filter2D(depth, -1, GRADIENT_D).astype(np.float32)
    lap_hor = np.abs(grad_l - grad_r)
    lap_ver = np.abs(grad_u - grad_d)
    lambda_map1, lambda_map2 = soft_min(lap_hor, base, 0)
    lambda_map3, lambda_map4 = soft_min(lap_ver, base, 1)

    eps = 1e-8
    thresh = base
    mask = lambda_map1 / (lambda_map2 + eps) > thresh
    lambda_map1[mask] = 1
    lambda_map2[mask] = 0
    mask = lambda_map2 / (lambda_map1 + eps) > thresh
    lambda_map1[mask] = 0
    lambda_map2[mask] = 1
    mask = lambda_map3 / (lambda_map4 + eps) > thresh
    lambda_map3[mask] = 1
    lambda_map4[mask] = 0
    mask = lambda_map4 / (lambda_map3 + eps) > thresh
    lambda_map3[mask] = 0
    lambda_map4[mask] = 1

    gu = lambda_map1 * grad_l + lambda_map2 * grad_r
    gv = lambda_map3 * grad_u + lambda_map4 * grad_d
    return gu.astype(np.float32), gv.astype(np.float32)


def mrf_optim(depth: np.ndarray, normal: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    depth = np.asarray(depth, dtype=np.float32)
    laplace = np.abs(cv2.filter2D(depth, -1, LAP_KER_ALPHA)).astype(np.float32)
    laplace_stack = np.array(
        (
            np.hstack((np.inf * np.ones((height, 1), dtype=np.float32), laplace[:, :-1])),
            np.hstack((laplace[:, 1:], np.inf * np.ones((height, 1), dtype=np.float32))),
            np.vstack((np.inf * np.ones((1, width), dtype=np.float32), laplace[:-1, :])),
            np.vstack((laplace[1:, :], np.inf * np.ones((1, width), dtype=np.float32))),
            laplace,
        ),
        dtype=np.float32,
    )
    best_loc = np.argmin(laplace_stack, axis=0).reshape(-1)
    channels = []
    for channel in range(3):
        values = normal[:, :, channel].astype(np.float32)
        stack = np.array(
            (
                np.hstack((np.zeros((height, 1), dtype=np.float32), values[:, :-1])),
                np.hstack((values[:, 1:], np.zeros((height, 1), dtype=np.float32))),
                np.vstack((np.zeros((1, width), dtype=np.float32), values[:-1, :])),
                np.vstack((values[1:, :], np.zeros((1, width), dtype=np.float32))),
                values,
            ),
            dtype=np.float32,
        ).reshape(5, -1)
        channels.append(stack[best_loc, np.arange(height * width)].reshape(height, width))
    return np.stack(channels, axis=-1).astype(np.float32)


def depth_to_normals(
    depth: np.ndarray,
    intrinsics: tuple[float, float, float, float],
    method: str,
    flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    height, width = depth.shape
    valid = (depth > 0) & np.isfinite(depth)
    safe_depth = depth.astype(np.float32).copy()
    safe_depth[~valid] = 0
    u_map = np.ones((height, 1), dtype=np.float32) * np.arange(1, width + 1, dtype=np.float32) - cx
    v_map = np.arange(1, height + 1, dtype=np.float32).reshape(height, 1) * np.ones((1, width), dtype=np.float32) - cy

    if method == "d2nt_basic":
        gu, gv = get_filter(safe_depth)
    else:
        gu, gv = get_dag_filter(safe_depth)

    normal = np.stack(
        (
            gu * fx,
            gv * fy,
            -(safe_depth + v_map * gv + u_map * gu),
        ),
        axis=-1,
    ).astype(np.float32)
    normal = vector_normalization(normal)
    if method == "d2nt_v3":
        normal = mrf_optim(safe_depth, normal)
    if flip:
        normal *= -1
    normal[~valid] = 0
    return normal.astype(np.float32), valid


def normal_vis(normal: np.ndarray) -> Image.Image:
    vis = ((normal.clip(-1, 1) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(vis, mode="RGB")


class LotusSVDDepth2Normal(torch.nn.Module):
    """Lotus plane-SVD depth-to-normal implementation.

    Mirrors EnVision-Research/Lotus utils/d2n/plane_svd.py with the VKITTI
    parameters used by utils/depth2normal.py: k=5, d=1, gamma=0.05,
    min_nghbr=4, d_min=1e-3, d_max=80.
    """

    def __init__(
        self,
        *,
        d_min: float = 1e-3,
        d_max: float = 80.0,
        k: int = 5,
        d: int = 1,
        min_nghbr: int = 4,
        gamma: float = 0.05,
    ) -> None:
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.k = k
        self.d = d
        self.min_nghbr = min_nghbr
        self.gamma = gamma
        self.pad = (k + (k - 1) * (d - 1)) // 2
        self.center_idx = (k * k - 1) // 2
        self.unfold = torch.nn.Unfold(kernel_size=(k, k), padding=self.pad, dilation=d)
        self.eigh_chunk_size = 8192

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = points.shape
        patches = self.unfold(points)
        matrix_a = patches.view(batch, 3, self.k * self.k, height, width)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1)

        valid_condition = (points[:, 2:3] > self.d_min) & (points[:, 2:3] < self.d_max)
        valid_condition = self.unfold(valid_condition.float())
        valid_condition = valid_condition.view(batch, 1, self.k * self.k, height, width)
        valid_condition = valid_condition.permute(0, 3, 4, 2, 1)

        center_depth = matrix_a[:, :, :, self.center_idx : self.center_idx + 1, 2:]
        valid_depth_diff = torch.abs(matrix_a[:, :, :, :, 2:] - center_depth) / center_depth.clamp_min(1e-12)
        valid_condition = valid_condition * (valid_depth_diff < self.gamma).float()

        ones = torch.ones_like(matrix_a[:, :, :, :, 0:1])
        matrix = torch.cat([matrix_a, ones], dim=-1)
        matrix = torch.where(valid_condition.repeat(1, 1, 1, 1, 4) > 0.5, matrix, torch.zeros_like(matrix))
        matrix_t = torch.transpose(matrix, 3, 4)

        matrix = matrix.reshape(-1, self.k * self.k, 4)
        matrix_t = matrix_t.reshape(-1, 4, self.k * self.k)
        ata = torch.bmm(matrix_t, matrix)
        eye = torch.eye(4, dtype=ata.dtype, device=ata.device).unsqueeze(0)
        ata = ata + eye * 1e-6
        normals = []
        for chunk in ata.split(self.eigh_chunk_size, dim=0):
            _eig_val, eig_vec = torch.linalg.eigh(chunk)
            normals.append(eig_vec[:, :3, 0])
        normal = torch.cat(normals, dim=0)
        normal = normal.view(batch, height, width, 3).permute(0, 3, 1, 2).contiguous()
        normal = F.normalize(normal, p=2.0, dim=1, eps=1e-12)

        flip = torch.sign(torch.sum(normal * points, dim=1, keepdim=True))
        normal = normal * flip

        valid_center = valid_condition[:, :, :, self.center_idx, 0].unsqueeze(1)
        valid_neighbors = torch.sum(valid_condition[..., 0], dim=3).unsqueeze(1) >= self.min_nghbr
        valid_normal = torch.norm(normal, p=2, dim=1, keepdim=True) > 0.5
        valid_mask = valid_center.bool() & valid_neighbors.bool() & valid_normal.bool()
        return normal, valid_mask


def lotus_points_from_depth_batch(
    depths: torch.Tensor,
    intrinsics: list[tuple[float, float, float, float]],
) -> torch.Tensor:
    batch, _, height, width = depths.shape
    device = depths.device
    dtype = depths.dtype
    u = torch.arange(width, dtype=dtype, device=device).view(1, 1, width).expand(batch, height, width)
    v = torch.arange(height, dtype=dtype, device=device).view(1, height, 1).expand(batch, height, width)
    fx = torch.tensor([item[0] for item in intrinsics], dtype=dtype, device=device).view(batch, 1, 1)
    fy = torch.tensor([item[1] for item in intrinsics], dtype=dtype, device=device).view(batch, 1, 1)
    cx = torch.tensor([item[2] for item in intrinsics], dtype=dtype, device=device).view(batch, 1, 1)
    cy = torch.tensor([item[3] for item in intrinsics], dtype=dtype, device=device).view(batch, 1, 1)
    z = depths[:, 0]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack((x, y, z), dim=1)


def process_lotus_svd_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not tasks:
        return []
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = LotusSVDDepth2Normal().to(device).eval()
    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))

    for start in range(0, len(tasks), batch_size):
        batch_tasks = tasks[start : start + batch_size]
        need_compute = [
            task
            for task in batch_tasks
            if not Path(task["normal_path"]).is_file()
            or not Path(task["mask_path"]).is_file()
            or (task["normal_vis_path"] and not Path(task["normal_vis_path"]).is_file())
        ]
        if need_compute:
            depths_np = [load_depth(Path(task["depth_path"]), task["depth_scale"]) for task in need_compute]
            depths = torch.from_numpy(np.stack(depths_np, axis=0)).float().unsqueeze(1).to(device)
            points = lotus_points_from_depth_batch(depths, [task["intrinsics"] for task in need_compute])
            with torch.no_grad():
                normals, valid_masks = model(points)
            if not need_compute[0]["flipped"]:
                normals = -normals
            normals_np = normals.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
            valid_np = valid_masks[:, 0].detach().cpu().numpy().astype(bool)
            for task, normal, valid in zip(need_compute, normals_np, valid_np):
                normal[~valid] = 0
                normal_path = Path(task["normal_path"])
                mask_path = Path(task["mask_path"])
                normal_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(normal_path, normal)
                Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(mask_path)
                if task["normal_vis_path"]:
                    vis_path = Path(task["normal_vis_path"])
                    vis_path.parent.mkdir(parents=True, exist_ok=True)
                    normal_vis(normal).save(vis_path)

        for task in batch_tasks:
            vis_path = Path(task["normal_vis_path"]) if task["normal_vis_path"] else None
            rows.append(
                {
                    "scene": task["scene"],
                    "variant": task["variant"],
                    "camera": task["camera"],
                    "frame": task["frame"],
                    "rgb_path": task["rgb_path"],
                    "depth_path": task["depth_path"],
                    "normal_path": task["normal_path"],
                    "mask_path": task["mask_path"],
                    "normal_vis_path": str(vis_path) if vis_path else "",
                    "fx": task["intrinsics"][0],
                    "fy": task["intrinsics"][1],
                    "cx": task["intrinsics"][2],
                    "cy": task["intrinsics"][3],
                    "method": task["method"],
                    "flipped": task["flipped"],
                }
            )
        if (start + len(batch_tasks)) % 100 == 0 or start + len(batch_tasks) == len(tasks):
            print(f"processed {start + len(batch_tasks)}/{len(tasks)}", flush=True)
    return rows


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


def process_sample(task: dict[str, Any]) -> dict[str, Any]:
    depth_path = Path(task["depth_path"])
    normal_path = Path(task["normal_path"])
    mask_path = Path(task["mask_path"])
    vis_path = Path(task["normal_vis_path"]) if task["normal_vis_path"] else None
    should_compute = not normal_path.is_file() or not mask_path.is_file() or (vis_path is not None and not vis_path.is_file())
    if should_compute:
        depth = load_depth(depth_path, task["depth_scale"])
        normal, valid = depth_to_normals(depth, task["intrinsics"], task["method"], task["flipped"])
        normal_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(normal_path, normal)
        Image.fromarray((valid.astype(np.uint8) * 255), mode="L").save(mask_path)
        if vis_path is not None:
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            normal_vis(normal).save(vis_path)
    row = {
        "scene": task["scene"],
        "variant": task["variant"],
        "camera": task["camera"],
        "frame": task["frame"],
        "rgb_path": task["rgb_path"],
        "depth_path": task["depth_path"],
        "normal_path": str(normal_path),
        "mask_path": str(mask_path),
        "normal_vis_path": str(vis_path) if vis_path else "",
        "fx": task["intrinsics"][0],
        "fy": task["intrinsics"][1],
        "cx": task["intrinsics"][2],
        "cy": task["intrinsics"][3],
        "method": task["method"],
        "flipped": task["flipped"],
    }
    return row


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    depth_root = resolve_vkitti2_root(args.root, "depth")
    rgb_root = resolve_vkitti2_root(args.root, "rgb")
    textgt_root = resolve_vkitti2_root(args.root, "textgt")

    intrinsics = load_intrinsics(textgt_root)
    depth_files = sorted(depth_root.glob("Scene*/**/frames/depth/Camera_*/*.png"))
    if args.max_samples > 0:
        depth_files = depth_files[: args.max_samples]
    if args.num_shards > 1:
        depth_files = depth_files[args.shard_index :: args.num_shards]
    if not depth_files:
        raise FileNotFoundError(f"No VKITTI2 depth PNG files found under {depth_root}")

    normal_dir = args.out_dir / "normals"
    mask_dir = args.out_dir / "masks"
    vis_dir = args.out_dir / "normal_vis"
    normal_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[dict[str, Any]] = []
    for depth_path in depth_files:
        scene, variant = scene_variant(depth_path)
        camera = camera_name(depth_path)
        frame = frame_index(depth_path)
        k = intrinsics.get((scene, variant, camera, frame), default_intrinsics(1242, 375))

        rel = Path(scene) / variant / camera / f"{frame:05d}"
        normal_path = normal_dir / rel.with_suffix(".npy")
        mask_path = mask_dir / rel.with_suffix(".png")
        vis_path = None
        if args.save_vis:
            vis_path = vis_dir / rel.with_suffix(".png")

        rgb_path = corresponding_rgb(depth_path, rgb_root)
        tasks.append(
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
                "intrinsics": k,
                "depth_scale": args.depth_scale,
                "method": args.method,
                "flipped": not args.no_flip,
            }
        )

    rows: list[dict[str, Any]] = []
    if args.method == "lotus_svd":
        rows = process_lotus_svd_tasks(tasks, args)
    elif args.workers <= 1:
        for idx, task in enumerate(tasks):
            rows.append(process_sample(task))
            if (idx + 1) % 100 == 0:
                print(f"processed {idx + 1}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for idx, row in enumerate(executor.map(process_sample, tasks, chunksize=16)):
                rows.append(row)
                if (idx + 1) % 100 == 0:
                    print(f"processed {idx + 1}/{len(tasks)}", flush=True)

    manifest = args.out_dir / (
        f"manifest.shard{args.shard_index:05d}-of-{args.num_shards:05d}.jsonl"
        if args.num_shards > 1
        else "manifest.jsonl"
    )
    rows.sort(key=lambda row: (row["scene"], row["variant"], row["camera"], row["frame"]))
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} samples to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
