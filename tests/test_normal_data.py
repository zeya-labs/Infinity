from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import infinity.normal_estimation.data as normal_data
from infinity.normal_estimation.data import HypersimNormalDataset, VKITTI2NormalDataset


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


if __name__ == "__main__":
    unittest.main()
