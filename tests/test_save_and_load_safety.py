from __future__ import annotations

import unittest
from pathlib import Path


class SaveAndLoadSafetyTest(unittest.TestCase):
    def test_checkpoint_auto_sync_does_not_shell_join_paths(self) -> None:
        text = Path("infinity/utils/save_and_load.py").read_text(encoding="utf-8")
        self.assertIn('cmd = ["cp", "-r", source_filename, target_filename]', text)
        self.assertIn("subprocess.Popen(cmd, bufsize=-1)", text)
        self.assertNotIn("cp -r {source_filename}", text)
        self.assertNotIn("shell=True", text)


if __name__ == "__main__":
    unittest.main()
