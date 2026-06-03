from __future__ import annotations

import numpy as np


def read_hdf5(path: str) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return handle["dataset"][:]


def create_uv_mesh(height: int, width: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(0, height, dtype=float),
        np.arange(0, width, dtype=float),
        indexing="ij",
    )
    meshgrid = np.stack((x, y))
    ones = np.ones((1, height * width), dtype=float)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)


def align_normal(normal: np.ndarray, depth: np.ndarray, intrinsics: list[float], height: int, width: int) -> np.ndarray:
    """Hypersim normals are not consistently oriented; flip them against the camera rays."""

    intrinsics_matrix = np.array(
        [
            [intrinsics[0], 0.0, intrinsics[1]],
            [0.0, intrinsics[2], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ]
    )
    inv_intrinsics = np.linalg.inv(intrinsics_matrix)
    xy = create_uv_mesh(height, width)
    points = np.matmul(inv_intrinsics[:3, :3], xy).reshape(3, height, width)
    points = depth * points
    points = points.transpose((1, 2, 0))

    orient_mask = np.sum(normal * points, axis=2) > 0
    normal[orient_mask] *= -1
    return normal


def read_depth_normal_hypersim(
    depth_path: str,
    normal_path: str,
    intrinsics: list[float],
    metric_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    depth = read_hdf5(depth_path).astype(np.float32)
    depth[depth > 60000] = 0
    depth = depth / metric_scale

    normal = read_hdf5(normal_path).astype(np.float32)
    height, width = normal.shape[:2]
    # Hypersim stores normals in (x right, y up, z backward)
    normal[:, :, 1:] *= -1
    normal = align_normal(normal, depth, intrinsics, height, width)
    normal /= np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5
    normal[:, :, 1:] *= -1
    normal[:, :, 0] *= -1
    return depth, normal
