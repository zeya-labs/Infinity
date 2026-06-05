from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from infinity.utils.save_and_load import _atomic_torch_save


class SaveAndLoadSafetyTest(unittest.TestCase):
    def test_checkpoint_auto_sync_does_not_shell_join_paths(self) -> None:
        text = Path("infinity/utils/save_and_load.py").read_text(encoding="utf-8")
        self.assertIn('cmd = ["cp", "-r", source_filename, target_filename]', text)
        self.assertIn("subprocess.Popen(cmd, bufsize=-1)", text)
        self.assertIn("_atomic_torch_save({", text)
        self.assertNotIn("cp -r {source_filename}", text)
        self.assertNotIn("shell=True", text)

    def test_atomic_torch_save_uses_replace(self) -> None:
        target_path = "/tmp/infinity-test/checkpoint.pth"
        with (
            mock.patch("infinity.utils.save_and_load.os.makedirs") as makedirs,
            mock.patch("infinity.utils.save_and_load.torch.save") as torch_save,
            mock.patch("infinity.utils.save_and_load.os.replace") as replace,
            mock.patch("infinity.utils.save_and_load.os.getpid", return_value=123),
        ):
            _atomic_torch_save({"step": 1}, target_path)

        makedirs.assert_called_once_with("/tmp/infinity-test", exist_ok=True)
        tmp_path = "/tmp/infinity-test/.checkpoint.pth.tmp.123"
        torch_save.assert_called_once_with({"step": 1}, tmp_path)
        replace.assert_called_once_with(tmp_path, target_path)

    def test_atomic_torch_save_removes_temp_file_on_failure(self) -> None:
        target_path = "/tmp/infinity-test/checkpoint.pth"
        tmp_path = "/tmp/infinity-test/.checkpoint.pth.tmp.123"
        with (
            mock.patch("infinity.utils.save_and_load.os.makedirs"),
            mock.patch("infinity.utils.save_and_load.torch.save", side_effect=RuntimeError("disk full")),
            mock.patch("infinity.utils.save_and_load.os.path.exists", return_value=True),
            mock.patch("infinity.utils.save_and_load.os.remove") as remove,
            mock.patch("infinity.utils.save_and_load.os.replace") as replace,
            mock.patch("infinity.utils.save_and_load.os.getpid", return_value=123),
        ):
            with self.assertRaisesRegex(RuntimeError, "disk full"):
                _atomic_torch_save({"step": 1}, target_path)

        remove.assert_called_once_with(tmp_path)
        replace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
