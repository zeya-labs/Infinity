from __future__ import annotations

import unittest
from pathlib import Path


class DistSafetyTest(unittest.TestCase):
    def test_backup_stream_log_link_does_not_shell_out(self) -> None:
        text = Path("infinity/utils/dist.py").read_text(encoding="utf-8")
        self.assertIn("def _link_run_trial_log", text)
        self.assertIn("os.path.isdir(run_trial_dir)", text)
        self.assertIn("os.symlink(fname, link_name)", text)
        self.assertIn("_link_run_trial_log(fname)", text)
        self.assertIn("INFINITY_RUN_TRIAL_DIR", text)
        self.assertIn('os.path.join("/opt", "tiger", "run_trial")', text)
        self.assertIn("failed to link log", text)
        self.assertIn("file=sys.stderr", text)
        self.assertNotIn("os.system", text)
        self.assertNotIn("ln -s", text)
        self.assertNotIn("/opt/tiger", text)


if __name__ == "__main__":
    unittest.main()
