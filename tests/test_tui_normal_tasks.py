from __future__ import annotations

import unittest
from pathlib import Path

import tui
from infinity.normal_estimation.defaults import (
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
        self.assertNotIn("--token-cache-" + "memory", cmd)

    def test_normal_tokenizer_uses_mixed_dataset_defaults(self) -> None:
        values = task_defaults("训练法线 Tokenizer")
        cmd = tui.build_train_tokenizer(values)
        self.assert_flag_value(cmd, "--train-datasets", DEFAULT_NORMAL_TRAIN_DATASETS)
        self.assert_flag_value(cmd, "--train-dataset-weights", DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS)
        self.assert_flag_value(cmd, "--vkitti2-root", DEFAULT_VKITTI2_ROOT)

    def test_tui_does_not_embed_local_vepfs_literal(self) -> None:
        text = Path(tui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/root/vepfs", text)


if __name__ == "__main__":
    unittest.main()
