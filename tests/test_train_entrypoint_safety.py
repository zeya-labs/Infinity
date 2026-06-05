from __future__ import annotations

import unittest
from pathlib import Path


class TrainEntrypointSafetyTest(unittest.TestCase):
    def test_train_wait_file_and_hdfs_helpers_do_not_use_shell(self) -> None:
        text = Path("train.py").read_text(encoding="utf-8")
        self.assertIn("def _touch_file", text)
        self.assertIn("def _remove_path", text)
        self.assertIn("def _hdfs_get_all", text)
        self.assertIn("def _print_rank_error_once", text)
        self.assertIn('["hdfs", "dfs", "-get", source_glob, target_dir]', text)
        self.assertNotIn("misc.os_system", text)
        self.assertNotIn("rm -rf", text)
        self.assertNotIn("shell=True", text)
        self.assertNotIn("except:", text)


if __name__ == "__main__":
    unittest.main()
