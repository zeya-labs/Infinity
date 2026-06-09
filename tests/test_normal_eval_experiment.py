import unittest

import numpy as np
import torch

from tools.normal_eval_experiment import (
    convert_target_to_eval,
    convert_prediction_to_eval,
    eval_normal_to_display_rgb,
    sample_id_from_metadata,
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

    def test_omnidata_prediction_convention_is_identity(self) -> None:
        prediction = torch.tensor([[[[-0.2]], [[0.3]], [[-0.4]]]], dtype=torch.float32)
        converted = convert_prediction_to_eval(prediction, "omnidata_v2")

        expected = prediction / torch.linalg.norm(prediction, dim=1, keepdim=True)
        self.assertTrue(torch.allclose(converted, expected))

    def test_eval_display_matches_infinity_normal_visualization(self) -> None:
        normal = np.asarray([[[1.0, -1.0, 0.0]]], dtype=np.float32)
        rgb = eval_normal_to_display_rgb(normal)

        np.testing.assert_array_equal(rgb[0, 0], np.asarray([0, 0, 127], dtype=np.uint8))

    def test_dsine_eval_targets_use_dsine_output_convention(self) -> None:
        normal = torch.tensor([[[[0.2]], [[-0.3]], [[0.4]]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "scannet"), convert_prediction_to_eval(normal, "dsine")))
        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "ibims"), convert_prediction_to_eval(normal, "dsine")))
        self.assertTrue(torch.allclose(convert_target_to_eval(normal, "sintel"), convert_prediction_to_eval(normal, "dsine")))


if __name__ == "__main__":
    unittest.main()
