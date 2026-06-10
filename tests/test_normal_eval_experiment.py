import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.normal_eval_experiment import (
    convert_target_to_eval,
    convert_prediction_to_eval,
    eval_normal_to_display_rgb,
    sample_id_from_metadata,
    write_canonical_predictions,
)


class NormalEvalExperimentTest(unittest.TestCase):
    def test_hypersim_eval_sample_id_includes_scene_and_camera(self) -> None:
        first = sample_id_from_metadata(
            0,
            {
                "dataset": "hypersim",
                "stem": "frame.0000.tonemap",
                "image_path": "/data/hypersim/ai_003_010/images/scene_cam_00_final_preview/frame.0000.tonemap.jpg",
            },
        )
        second = sample_id_from_metadata(
            1,
            {
                "dataset": "hypersim",
                "stem": "frame.0000.tonemap",
                "image_path": "/data/hypersim/ai_004_001/images/scene_cam_01_final_preview/frame.0000.tonemap.jpg",
            },
        )

        self.assertEqual(first, "ai_003_010_scene_cam_00_final_preview_frame.0000.tonemap")
        self.assertEqual(second, "ai_004_001_scene_cam_01_final_preview_frame.0000.tonemap")
        self.assertNotEqual(first, second)

    def test_non_hypersim_eval_sample_id_keeps_existing_stem(self) -> None:
        self.assertEqual(sample_id_from_metadata(7, {"dataset": "nyuv2", "stem": "val_000007"}), "val_000007")

    def test_ours_prediction_convention_is_identity(self) -> None:
        prediction = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_prediction_to_eval(prediction, "ours")

        expected = prediction / torch.linalg.norm(prediction, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_omnidata_adapter_prediction_convention_is_identity(self) -> None:
        prediction = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_prediction_to_eval(prediction, "omnidata_v2")

        expected = prediction / torch.linalg.norm(prediction, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_marigold_prediction_flips_x_to_canonical(self) -> None:
        prediction = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_prediction_to_eval(prediction, "marigold")

        expected = torch.tensor([[[[-0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        expected = expected / torch.linalg.norm(expected, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_nyuv2_target_converts_to_left_up_backward(self) -> None:
        target = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)
        converted = convert_target_to_eval(target, "nyuv2")

        expected = torch.tensor([[[[-0.2]], [[0.4]], [[0.3]]]], dtype=torch.float32)
        expected = expected / torch.linalg.norm(expected, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_eval_display_is_direct_canonical_visualization(self) -> None:
        normal = np.asarray([[[1.0, -1.0, 0.0]]], dtype=np.float32)
        rgb = eval_normal_to_display_rgb(normal)

        np.testing.assert_array_equal(rgb[0, 0], np.asarray([255, 0, 127], dtype=np.uint8))

    def test_write_canonical_predictions_saves_eval_npy_and_direct_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            method_dir = Path(tmp) / "marigold"
            pred_dir = method_dir / "normals_npy"
            pred_dir.mkdir(parents=True)
            raw = np.asarray([[[0.2, -0.3, 0.4]]], dtype=np.float32)
            np.save(pred_dir / "sample_normals.npy", raw)
            mask_path = Path(tmp) / "sample_mask.png"
            Image.fromarray(np.asarray([[255]], dtype=np.uint8)).save(mask_path)

            count = write_canonical_predictions(
                method_dir,
                ["sample"],
                "marigold",
                [{"id": "sample", "mask": str(mask_path)}],
            )

            expected = np.asarray([[[-0.2, -0.3, 0.4]]], dtype=np.float32)
            expected = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
            canonical_dir = method_dir / "canonical_predictions"
            self.assertEqual(count, 1)
            np.testing.assert_allclose(np.load(canonical_dir / "sample_normal.npy"), expected, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(
                np.asarray(Image.open(canonical_dir / "sample_normal.png")),
                eval_normal_to_display_rgb(expected, np.asarray([[True]])),
            )
            metrics = json.loads((method_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["canonical_predictions"]["num_png"], 1)
            self.assertEqual(metrics["canonical_predictions"]["num_npy"], 1)

    def test_dsine_eval_targets_use_dsine_output_convention(self) -> None:
        normal = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "scannet"), convert_prediction_to_eval(normal, "dsine")))
        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "ibims"), convert_prediction_to_eval(normal, "dsine")))
        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "sintel"), convert_prediction_to_eval(normal, "dsine")))


if __name__ == "__main__":
    unittest.main()
