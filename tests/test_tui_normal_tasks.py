from __future__ import annotations

import unittest
from pathlib import Path

import tui
from infinity.normal_estimation.defaults import (
    DEFAULT_NORMAL_ESTIMATION_CKPT,
    DEFAULT_NORMAL_TOKENIZER_CKPT,
    DEFAULT_NORMAL_TRAIN_DATASETS,
    DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS,
    DEFAULT_VKITTI2_ROOT,
)


def task_defaults(title: str) -> dict[str, str]:
    task = next(item for item in tui.TASKS if item.title == title)
    return {field.key: field.default for field in task.fields}


class TuiNormalTaskTest(unittest.TestCase):
    def assert_flag_value(self, cmd: list[str], flag: str, expected: str) -> None:
        self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index(flag) + 1], expected)

    def test_normal_estimation_uses_mixed_dataset_defaults(self) -> None:
        values = task_defaults("训练 RGB 到 Normal")
        cmd = tui.build_train_normal(values)
        self.assert_flag_value(cmd, "--train-datasets", DEFAULT_NORMAL_TRAIN_DATASETS)
        self.assert_flag_value(cmd, "--train-dataset-weights", DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS)
        self.assert_flag_value(cmd, "--vkitti2-root", DEFAULT_VKITTI2_ROOT)
        self.assert_flag_value(cmd, "--lr", "1e-5")
        self.assert_flag_value(cmd, "--word-head-lr", "2e-5")
        self.assert_flag_value(cmd, "--image-word-lr", "5e-5")
        self.assert_flag_value(cmd, "--normal-task-lr", "1e-4")
        self.assert_flag_value(cmd, "--epochs", "5")
        self.assert_flag_value(cmd, "--ar-eval-nyuv2-root", "data/NYUv2/hf-parquet/tanganke/nyuv2/data")
        self.assert_flag_value(cmd, "--ar-eval-nyuv2-samples", "32")
        self.assert_flag_value(cmd, "--max-val-samples", "512")
        self.assert_flag_value(cmd, "--train-normal-metrics-every", "100")
        self.assert_flag_value(cmd, "--image-log-every", "200")
        self.assert_flag_value(cmd, "--save-every-steps", "500")
        self.assertNotIn("--hypersim-filter-depth-nan", cmd)
        self.assertNotIn("--token-cache-" + "memory", cmd)

    def test_normal_estimation_all_6e_5_lr_ablation(self) -> None:
        values = task_defaults("训练 RGB 到 Normal")
        values["lr_ablation"] = "all_6e-5"
        cmd = tui.build_train_normal(values)
        self.assert_flag_value(cmd, "--lr", "6e-5")
        self.assert_flag_value(cmd, "--min-lr", "6e-6")
        self.assert_flag_value(cmd, "--word-head-lr", "6e-5")
        self.assert_flag_value(cmd, "--image-word-lr", "6e-5")
        self.assert_flag_value(cmd, "--normal-task-lr", "6e-5")

    def test_normal_tokenizer_uses_mixed_dataset_defaults(self) -> None:
        values = task_defaults("训练法线 Tokenizer")
        cmd = tui.build_train_tokenizer(values)
        self.assert_flag_value(cmd, "--train-datasets", DEFAULT_NORMAL_TRAIN_DATASETS)
        self.assert_flag_value(cmd, "--train-dataset-weights", DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS)
        self.assert_flag_value(cmd, "--vkitti2-root", DEFAULT_VKITTI2_ROOT)
        self.assert_flag_value(cmd, "--lr", "6e-5")
        self.assert_flag_value(cmd, "--min-lr", "6e-6")
        self.assert_flag_value(cmd, "--epochs", "5")
        self.assertNotIn("--hypersim-filter-depth-nan", cmd)

    def test_normal_eval_uses_shared_defaults_and_all_baselines(self) -> None:
        values = task_defaults("Normal Eval 实验")
        cmd = tui.build_normal_baseline_compare(values)

        self.assertIn("tools/normal_eval_experiment.py", cmd)
        self.assert_flag_value(cmd, "--ours-checkpoint", DEFAULT_NORMAL_ESTIMATION_CKPT)
        self.assert_flag_value(cmd, "--normal-tokenizer-ckpt", DEFAULT_NORMAL_TOKENIZER_CKPT)
        for method in (
            "ours",
            "marigold",
            "geowizard",
            "stablenormal",
            "lotusg",
            "dsine",
            "metric3dv2",
            "omnidata_v2",
            "marigold_e2eft",
            "lotusd",
        ):
            self.assertIn(method, cmd)

    def test_tui_does_not_embed_local_vepfs_literal(self) -> None:
        text = Path(tui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/root/vepfs", text)


if __name__ == "__main__":
    unittest.main()
