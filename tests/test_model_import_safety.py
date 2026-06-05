from __future__ import annotations

import unittest
from pathlib import Path


class ModelImportSafetyTest(unittest.TestCase):
    def test_optional_fused_op_import_only_catches_import_error(self) -> None:
        text = Path("infinity/models/infinity.py").read_text(encoding="utf-8")
        self.assertIn("except ImportError:", text)
        self.assertNotIn("from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm\nexcept:", text)

    def test_vqgan_memory_fallbacks_only_catch_runtime_error(self) -> None:
        flux = Path("infinity/models/bsq_vae/flux_vqgan.py").read_text(encoding="utf-8")
        conv = Path("infinity/models/bsq_vae/conv.py").read_text(encoding="utf-8")
        self.assertIn("except RuntimeError:", flux)
        self.assertIn("except RuntimeError:", conv)
        self.assertNotIn("except:\n", flux)
        self.assertNotIn("except:\n", conv)


if __name__ == "__main__":
    unittest.main()
