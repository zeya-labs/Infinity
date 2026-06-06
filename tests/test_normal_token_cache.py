from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from infinity.normal_estimation.token_cache import token_cache_sample_key
from infinity.utils.torch_io import atomic_torch_save
from tools.train_normal_estimation import batch_dataset_name, load_token_cache_batch, save_token_cache_batch


class NormalTokenCacheTest(unittest.TestCase):
    def _metadata(self, index: int = 7) -> dict:
        return {
            "dataset": "hypersim",
            "partition": "train",
            "index": index,
            "image_path": f"/data/hypersim/rgb/{index:05d}.jpg",
            "normal_path": f"/data/hypersim/normal/{index:05d}.hdf5",
            "target_size": [240, 320],
        }

    def test_vkitti_manifest_dir_disambiguates_relative_paths(self) -> None:
        metadata = {
            "dataset": "vkitti2",
            "partition": "train",
            "index": 7,
            "image_path": "rgb/00007.png",
            "normal_path": "normal/00007.npy",
            "target_size": [240, 320],
        }
        first = dict(metadata, manifest_dir="/data/vkitti-a")
        second = dict(metadata, manifest_dir="/data/vkitti-b")

        self.assertNotEqual(
            token_cache_sample_key(first, "sig"),
            token_cache_sample_key(second, "sig"),
        )

    def test_hypersim_key_ignores_absent_manifest_dir(self) -> None:
        metadata = self._metadata()

        self.assertEqual(
            token_cache_sample_key(metadata, "sig"),
            token_cache_sample_key(dict(metadata), "sig"),
        )

    def test_cache_miss_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertIsNone(
                load_token_cache_batch(
                    Path(tmpdir),
                    [self._metadata()],
                    "sig",
                    torch.device("cpu"),
                )
            )

    def test_cache_hit_loads_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = self._metadata()
            payload = {
                "rgb_prefix_blc": torch.tensor([[1.0, 2.0]]),
                "x_blc_without_prefix": torch.tensor([[3.0, 4.0]]),
                "gt_bl": torch.tensor([5, 6]),
                "raw_features": torch.tensor([[[7.0]]]),
            }
            atomic_torch_save(payload, cache_dir / token_cache_sample_key(metadata, "sig"))

            rgb_prefix_blc, x_blc_without_prefix, gt_bl, raw_features = load_token_cache_batch(
                cache_dir,
                [metadata],
                "sig",
                torch.device("cpu"),
            )

            torch.testing.assert_close(rgb_prefix_blc, payload["rgb_prefix_blc"].unsqueeze(0))
            torch.testing.assert_close(x_blc_without_prefix, payload["x_blc_without_prefix"].unsqueeze(0))
            torch.testing.assert_close(gt_bl, payload["gt_bl"].unsqueeze(0))
            torch.testing.assert_close(raw_features, payload["raw_features"].unsqueeze(0))

    def test_save_token_cache_batch_writes_disk_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            metadata = [self._metadata(1), self._metadata(2)]

            save_token_cache_batch(
                cache_dir,
                metadata,
                "sig",
                rgb_prefix_blc=torch.arange(4, dtype=torch.float32).reshape(2, 2),
                x_blc_without_prefix=torch.arange(4, 8, dtype=torch.float32).reshape(2, 2),
                gt_bl=torch.arange(4, dtype=torch.long).reshape(2, 2),
                raw_features=torch.arange(2, dtype=torch.float32).reshape(2, 1),
            )

            for item in metadata:
                self.assertTrue((cache_dir / token_cache_sample_key(item, "sig")).is_file())

    def test_memory_cache_cli_is_removed(self) -> None:
        text = Path("tools/train_normal_estimation.py").read_text(encoding="utf-8")
        self.assertNotIn("--token-cache-" + "memory", text)
        self.assertNotIn("TOKEN_CACHE_" + "MEMORY", text)

    def test_batch_dataset_name(self) -> None:
        self.assertEqual(batch_dataset_name({"metadata": [self._metadata(1), self._metadata(2)]}), "hypersim")
        self.assertEqual(
            batch_dataset_name({"metadata": [self._metadata(1), {**self._metadata(2), "dataset": "vkitti2"}]}),
            "mixed",
        )


if __name__ == "__main__":
    unittest.main()
