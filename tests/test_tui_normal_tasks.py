from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import tui
from infinity.normal_estimation.defaults import (
    DEFAULT_NORMAL_TOKENIZER_CKPT,
    DEFAULT_NORMAL_TRAIN_DATASET_WEIGHTS,
    DEFAULT_NORMAL_TRAIN_DATASETS,
    DEFAULT_VKITTI2_MAX_INVALID_RATIO,
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
        self.assert_flag_value(cmd, "--vkitti2-max-invalid-ratio", f"{DEFAULT_VKITTI2_MAX_INVALID_RATIO:g}")
        self.assert_flag_value(cmd, "--lr", "1e-5")
        self.assert_flag_value(cmd, "--word-head-lr", "2e-5")
        self.assert_flag_value(cmd, "--image-word-lr", "5e-5")
        self.assert_flag_value(cmd, "--normal-task-lr", "1e-4")
        self.assert_flag_value(cmd, "--normal-token-layout", "prefix")
        self.assert_flag_value(cmd, "--epochs", "5")
        self.assert_flag_value(cmd, "--ar-eval-nyuv2-root", "data/NYUv2/hf-parquet/tanganke/nyuv2/data")
        self.assert_flag_value(cmd, "--ar-eval-nyuv2-samples", "32")
        self.assert_flag_value(cmd, "--max-val-samples", "512")
        self.assert_flag_value(cmd, "--train-normal-metrics-every", "10")
        self.assert_flag_value(cmd, "--image-log-every", "200")
        self.assert_flag_value(cmd, "--save-every-steps", "100")
        self.assertIn("--hypersim-filter-depth-nan", cmd)
        self.assertNotIn("--token-cache-" + "memory", cmd)

    def test_normal_estimation_can_select_interleaved_token_layout(self) -> None:
        values = task_defaults("训练 RGB 到 Normal")
        values["normal_token_layout"] = "interleaved"
        cmd = tui.build_train_normal(values)
        self.assert_flag_value(cmd, "--normal-token-layout", "interleaved")

    def test_normal_estimation_strict_interleaved_source_uses_flex_attention(self) -> None:
        values = task_defaults("训练 RGB 到 Normal")
        values["normal_token_layout"] = "interleaved_source"
        values["normal_use_segmented_flash_attn"] = "1"
        cmd = tui.build_train_normal(values)
        self.assert_flag_value(cmd, "--normal-token-layout", "interleaved_source")
        self.assertIn("--normal-use-flex-attn", cmd)
        self.assertNotIn("--normal-use-segmented-flash-attn", cmd)

    def test_normal_estimation_can_disable_hypersim_nan_filter_flag(self) -> None:
        values = task_defaults("训练 RGB 到 Normal")
        values["hypersim_filter_depth_nan"] = "0"
        cmd = tui.build_train_normal(values)
        self.assertNotIn("--hypersim-filter-depth-nan", cmd)

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
        self.assert_flag_value(cmd, "--vkitti2-max-invalid-ratio", f"{DEFAULT_VKITTI2_MAX_INVALID_RATIO:g}")
        self.assert_flag_value(cmd, "--lr", "6e-5")
        self.assert_flag_value(cmd, "--min-lr", "6e-6")
        self.assert_flag_value(cmd, "--epochs", "5")
        self.assertIn("--hypersim-filter-depth-nan", cmd)

    def test_normal_eval_uses_shared_defaults_and_all_baselines(self) -> None:
        values = task_defaults("Normal Eval 实验")
        cmd = tui.build_normal_baseline_compare(values)

        self.assertIn("tools/normal_eval_experiment.py", cmd)
        self.assert_flag_value(cmd, "--ours-checkpoint", tui.normal_checkpoint_choices()[0])
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

    def test_normal_eval_baseline_only_omits_ours_flags(self) -> None:
        values = task_defaults("Normal Eval 实验")
        values["methods"] = "stablenormal"
        cmd = tui.build_single_normal_eval(values, "nyuv2", values["output_dir"])

        self.assertIn("stablenormal", cmd)
        self.assertNotIn("--ours-checkpoint", cmd)
        self.assertNotIn("--normal-tokenizer-ckpt", cmd)
        self.assertNotIn("--normal-vae-type", cmd)
        self.assertNotIn("--ours-seed", cmd)
        self.assertNotIn("--ours-kv-cache-fast", cmd)

    def test_tui_does_not_embed_local_vepfs_literal(self) -> None:
        text = Path(tui.__file__).read_text(encoding="utf-8")
        self.assertNotIn("/root/vepfs", text)

    def test_hf_upload_task_builds_checkpoint_upload_command(self) -> None:
        values = task_defaults("上传 HF checkpoint")
        values["repo_id"] = "user/model"
        values["checkpoint"] = "outputs/normal_estimation/latest/checkpoints/model.pth"
        values["path_in_repo"] = "checkpoints/model.pth"
        cmd = tui.build_upload_hf_checkpoint(values)

        self.assertIn("scripts/upload_hf_checkpoint.py", cmd)
        self.assertNotIn("--dry-run", cmd)
        self.assertIn("--create-repo", cmd)
        self.assertIn("--private", cmd)
        self.assert_flag_value(cmd, "--repo-id", "user/model")
        self.assert_flag_value(cmd, "--checkpoint", "outputs/normal_estimation/latest/checkpoints/model.pth")
        self.assert_flag_value(cmd, "--path-in-repo", "checkpoints/model.pth")
        self.assertNotIn("HF_TOKEN", cmd)

    def test_hf_upload_env_sets_token_and_proxy(self) -> None:
        with mock.patch.dict(tui.os.environ, {"HF_TOKEN": "test-token"}, clear=False):
            env = tui.hf_upload_env({})

        self.assertEqual(env["HF_TOKEN"], "test-token")
        self.assertEqual(env["http_proxy"], tui.DEFAULT_HF_UPLOAD_PROXY)
        self.assertEqual(env["HTTP_PROXY"], tui.DEFAULT_HF_UPLOAD_PROXY)
        self.assertEqual(env["https_proxy"], tui.DEFAULT_HF_UPLOAD_PROXY)
        self.assertEqual(env["HTTPS_PROXY"], tui.DEFAULT_HF_UPLOAD_PROXY)


if __name__ == "__main__":
    unittest.main()
