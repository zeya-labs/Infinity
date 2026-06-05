from __future__ import annotations

import unittest
from pathlib import Path


class TrainerCheckpointSafetyTest(unittest.TestCase):
    def test_optional_checkpoint_config_is_checked_before_access(self) -> None:
        text = Path("trainer.py").read_text(encoding="utf-8")
        config_pos = text.index("config: dict = state.pop('config', None)")
        guard_pos = text.index("if config is not None:", config_pos)
        first_get_pos = text.index("config.get('prog_it', 0)", config_pos)

        self.assertLess(guard_pos, first_get_pos)
        self.assertNotIn("config.get(k, None)", text)


if __name__ == "__main__":
    unittest.main()
