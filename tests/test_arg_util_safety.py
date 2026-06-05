from __future__ import annotations

import unittest
from pathlib import Path


class ArgUtilSafetyTest(unittest.TestCase):
    def test_git_branch_detection_does_not_use_shell(self) -> None:
        text = Path("infinity/utils/arg_util.py").read_text(encoding="utf-8")
        self.assertIn('["git", "symbolic-ref", "--short", "HEAD"]', text)
        self.assertIn('["git", "rev-parse", "HEAD"]', text)
        self.assertNotIn("git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD", text)
        self.assertNotIn("shell=True", text)

    def test_legacy_state_parser_does_not_use_eval(self) -> None:
        text = Path("infinity/utils/arg_util.py").read_text(encoding="utf-8")
        self.assertIn("ast.literal_eval", text)
        self.assertIn("json.loads", text)
        self.assertNotIn("eval('\\n'.join", text)

    def test_ready_node_cleanup_does_not_shell_rm(self) -> None:
        text = Path("infinity/utils/arg_util.py").read_text(encoding="utf-8")
        self.assertIn("def _remove_ready_node_markers", text)
        self.assertIn('glob.glob(os.path.join(directory, "ready-node*"))', text)
        self.assertIn("shutil.rmtree(path)", text)
        self.assertIn("os.remove(path)", text)
        self.assertNotIn("os.system", text)
        self.assertNotIn("rm -rf", text)

    def test_output_directory_creation_is_not_silently_swallowed(self) -> None:
        text = Path("infinity/utils/arg_util.py").read_text(encoding="utf-8")
        self.assertIn("def _ensure_directory", text)
        self.assertIn("_ensure_directory(args.bed)", text)
        self.assertIn("_ensure_directory(args.local_out_path)", text)
        self.assertNotIn("try: os.makedirs(args.bed", text)
        self.assertNotIn("except: pass", text)


if __name__ == "__main__":
    unittest.main()
