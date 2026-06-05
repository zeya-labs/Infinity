from __future__ import annotations

import unittest

from tools.train_normal_estimation import token_cache_sample_key


class NormalTokenCacheTest(unittest.TestCase):
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
        metadata = {
            "dataset": "hypersim",
            "partition": "train",
            "index": 7,
            "image_path": "/data/hypersim/rgb/00007.jpg",
            "normal_path": "/data/hypersim/normal/00007.hdf5",
            "target_size": [240, 320],
        }

        self.assertEqual(
            token_cache_sample_key(metadata, "sig"),
            token_cache_sample_key(dict(metadata), "sig"),
        )


if __name__ == "__main__":
    unittest.main()
