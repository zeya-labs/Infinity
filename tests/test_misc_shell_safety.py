from __future__ import annotations

import unittest
from pathlib import Path

import torch

from infinity.utils import misc


class MiscShellSafetyTest(unittest.TestCase):
    def test_misc_command_helpers_do_not_use_shell(self) -> None:
        text = Path("infinity/utils/misc.py").read_text(encoding="utf-8")
        self.assertIn("def _normalize_command", text)
        self.assertIn("shlex.split(cmd)", text)
        self.assertIn("subprocess.call(_normalize_command(cmd))", text)
        self.assertIn("subprocess.run(_normalize_command(cmd), stdout=subprocess.PIPE)", text)
        self.assertNotIn("shell=True", text)
        self.assertNotIn("functools.partial(subprocess.call", text)

    def test_echo_uses_python_print_instead_of_shell_echo(self) -> None:
        text = Path("infinity/utils/misc.py").read_text(encoding="utf-8")
        self.assertIn("def echo(info):", text)
        self.assertIn("print(", text)
        self.assertNotIn("os_system(f'echo", text)

    def test_optional_import_and_touch_errors_are_explicit(self) -> None:
        text = Path("infinity/utils/misc.py").read_text(encoding="utf-8")
        self.assertIn("except ImportError:", text)
        self.assertIn("tfio = None", text)
        self.assertIn("except OSError as exc:", text)
        self.assertIn("failed to touch", text)
        self.assertNotIn("except: pass", text)

    def test_get_param_for_log_returns_detached_statistics(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        for param in model[1].parameters():
            param.requires_grad_(False)

        stats = misc.get_param_for_log("toy", model.named_parameters())

        self.assertEqual({"toy/0.weight", "toy/0.bias"}, set(stats))
        for values in stats.values():
            self.assertEqual(2, len(values))
            self.assertTrue(all(isinstance(value, float) for value in values))


if __name__ == "__main__":
    unittest.main()
