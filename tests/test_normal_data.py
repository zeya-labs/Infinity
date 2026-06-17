from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import infinity.normal_estimation.data as normal_data
from PIL import Image

from infinity.normal_estimation.data import DSINEEvalNormalDataset, HypersimNormalDataset, VKITTI2NormalDataset


class NormalDataTest(unittest.TestCase):
    def test_vkitti2_manifest_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Manifest is empty"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)

    def test_vkitti2_manifest_reports_bad_json_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                '{"rgb_path": "rgb.png", "normal_path": "normal.npy", "mask_path": "mask.png"}\n{bad-json}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "line 2"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)

    def test_vkitti2_manifest_reports_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text('{"rgb_path": "rgb.png"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                VKITTI2NormalDataset(root=tmpdir, metadata_only=True)

    def test_vkitti2_filters_by_max_invalid_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            Image.new("RGB", (2, 2), (0, 0, 0)).save(root / "rgb.png")
            np.save(root / "normal0.npy", np.ones((2, 2, 3), dtype=np.float32))
            np.save(root / "normal1.npy", np.ones((2, 2, 3), dtype=np.float32))
            Image.fromarray(np.array([[255, 255], [255, 0]], dtype=np.uint8)).save(root / "mask0.png")
            Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(root / "mask1.png")
            records = [
                {"rgb_path": "rgb.png", "normal_path": "normal0.npy", "mask_path": "mask0.png"},
                {"rgb_path": "rgb.png", "normal_path": "normal1.npy", "mask_path": "mask1.png"},
            ]
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            dataset = VKITTI2NormalDataset(root=tmpdir, metadata_only=True, max_invalid_ratio=0.25)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.get_metadata(0)["normal_path"], "normal0.npy")
            self.assertAlmostEqual(dataset.get_metadata(0)["mask_invalid_ratio"], 0.25)

    def test_hypersim_filter_depth_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "final_train_split.csv"
            split.write_text(
                "scene_name,camera_name,frame_id,images,depth,normal,settings_camera_fov\n"
                "scene,cam,0,rgb0.jpg,depth0.hdf5,normal0.hdf5,1.0\n"
                "scene,cam,1,rgb1.jpg,depth1.hdf5,normal1.hdf5,1.0\n",
                encoding="utf-8",
            )

            original_read_hdf5 = normal_data.read_hdf5

            def fake_read_hdf5(path: str) -> np.ndarray:
                if path.endswith("depth0.hdf5"):
                    return np.array([[1.0, np.nan]], dtype=np.float32)
                return np.array([[1.0, 2.0]], dtype=np.float32)

            normal_data.read_hdf5 = fake_read_hdf5
            try:
                dataset = HypersimNormalDataset(root=tmpdir, partition="train", metadata_only=True, filter_depth_nan=True)
            finally:
                normal_data.read_hdf5 = original_read_hdf5

            self.assertEqual(len(dataset), 1)
            self.assertIn("depth1.hdf5", dataset.get_metadata(0)["depth_path"])

    def test_dsine_eval_dataset_discovers_scannet_style_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scene = Path(tmpdir) / "scannet" / "scene0000_00"
            scene.mkdir(parents=True)
            Image.new("RGB", (4, 3), (32, 64, 96)).save(scene / "000000_img.png")
            Image.new("RGB", (4, 3), (255, 127, 127)).save(scene / "000000_normal.png")

            dataset = DSINEEvalNormalDataset(root=tmpdir, dataset="scannet", metadata_only=True)

            self.assertEqual(len(dataset), 1)
            metadata = dataset.get_metadata(0)
            self.assertEqual(metadata["dataset"], "scannet")
            self.assertEqual(metadata["image_path"], "scene0000_00/000000_img.png")
            self.assertEqual(metadata["normal_path"], "scene0000_00/000000_normal.png")


if __name__ == "__main__":
    unittest.main()
