from __future__ import annotations

import subprocess
import sys
import textwrap
import unittest


class NormalImportTest(unittest.TestCase):
    def test_defaults_import_does_not_import_torch(self) -> None:
        code = textwrap.dedent(
            """
            import sys
            import infinity.normal_estimation.defaults  # noqa: F401
            raise SystemExit(1 if 'torch' in sys.modules else 0)
            """
        )
        result = subprocess.run([sys.executable, "-c", code], check=False)
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
