from __future__ import annotations

import unittest
from pathlib import Path


class DatasetBuildSafetyTest(unittest.TestCase):
    def test_rewrite_does_not_shell_join_sudo_commands(self) -> None:
        text = Path("infinity/dataset/build.py").read_text(encoding="utf-8")
        self.assertIn('subprocess.run(["sudo", "mv", tmp_file, file], check=True)', text)
        self.assertIn('subprocess.run(["sudo", "chown", f"{uname}:{gname}", file], check=True)', text)
        self.assertIn('subprocess.run(["sudo", "chmod", mode, file], check=True)', text)
        self.assertIn("tempfile.mkstemp", text)
        self.assertNotIn("sudo mv", text)
        self.assertNotIn("shell=True", text)

    def test_pwd_grp_import_has_numeric_fallback(self) -> None:
        text = Path("infinity/dataset/build.py").read_text(encoding="utf-8")
        self.assertIn("except ImportError:", text)
        self.assertIn("getpwuid = None", text)
        self.assertIn("getgrgid = None", text)
        self.assertIn("def _user_name", text)
        self.assertIn("def _group_name", text)
        self.assertIn("return str(uid)", text)
        self.assertIn("return str(gid)", text)
        self.assertNotIn("except:\n    pass", text)


if __name__ == "__main__":
    unittest.main()
