from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from infinity.utils import dist


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

    def test_finalize_resets_module_state(self) -> None:
        old_state = {
            "__rank": dist.__dict__["__rank"],
            "__local_rank": dist.__dict__["__local_rank"],
            "__world_size": dist.__dict__["__world_size"],
            "__device": dist.__dict__["__device"],
            "__rank_str_zfill": dist.__dict__["__rank_str_zfill"],
            "__initialized": dist.__dict__["__initialized"],
        }
        try:
            dist.__dict__["__rank"] = 3
            dist.__dict__["__local_rank"] = 1
            dist.__dict__["__world_size"] = 8
            dist.__dict__["__device"] = "cuda:1"
            dist.__dict__["__rank_str_zfill"] = "3"
            dist.__dict__["__initialized"] = True

            with mock.patch.object(dist.tdist, "destroy_process_group") as destroy_process_group:
                dist.finalize()

            destroy_process_group.assert_called_once_with()
            self.assertFalse(dist.initialized())
            self.assertEqual(dist.get_rank(), 0)
            self.assertEqual(dist.get_local_rank(), 0)
            self.assertEqual(dist.get_world_size(), 1)
            self.assertEqual(dist.get_device(), "cpu")
            self.assertEqual(dist.get_rank_str_zfill(), "0")
        finally:
            dist.__dict__.update(old_state)


if __name__ == "__main__":
    unittest.main()
