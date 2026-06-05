from __future__ import annotations

import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

from infinity.utils.swanlab_utils import build_swanlab_experiment_name, init_swanlab_run, swanlab_local_resume_patch


class SwanLabUtilsTest(unittest.TestCase):
    def test_build_experiment_name_uses_explicit_name(self) -> None:
        self.assertEqual(
            build_swanlab_experiment_name(Path("/tmp/out"), "manual", "train"),
            "manual",
        )

    def test_build_experiment_name_uses_output_parent_and_name(self) -> None:
        self.assertEqual(
            build_swanlab_experiment_name(Path("/tmp/runs/latest"), "", "train"),
            "train_runs_latest",
        )

    def test_disabled_init_does_not_import_swanlab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("infinity.utils.swanlab_utils.import_swanlab") as mocked_import:
                run = init_swanlab_run(
                    output_dir=Path(tmpdir),
                    enabled=False,
                    mode="cloud",
                    project="project",
                    workspace="",
                    experiment_name="",
                    job_type="job",
                    tags=[],
                    config={},
                )
        self.assertIsNone(run)
        mocked_import.assert_not_called()

    def test_malformed_state_file_is_ignored_and_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "run"
            output_dir.mkdir()
            state_path = output_dir / "swanlab_run.json"
            state_path.write_text("{not-json", encoding="utf-8")
            fake_run = SimpleNamespace(public=SimpleNamespace(run_id="new-run", run_dir=""))
            fake_swanlab = SimpleNamespace(init=mock.Mock(return_value=fake_run))
            logger = mock.Mock()

            with mock.patch("infinity.utils.swanlab_utils.import_swanlab", return_value=fake_swanlab):
                run = init_swanlab_run(
                    output_dir=output_dir,
                    enabled=True,
                    mode="cloud",
                    project="project",
                    workspace="",
                    experiment_name="",
                    job_type="job",
                    tags=[],
                    config={},
                    logger=logger,
                )

            self.assertIs(run, fake_run)
            logger.warning.assert_called_once()
            self.assertIn('"run_id": "new-run"', state_path.read_text(encoding="utf-8"))

    def test_local_resume_patch_skips_unsupported_new_swanlab_run_id_api(self) -> None:
        try:
            import swanlab.data.callbacker.local as swanlab_local  # noqa: F401
            import swanlab.data.run.main as swanlab_run_main
            import swankit.core.settings as swankit_settings
        except Exception as exc:
            raise unittest.SkipTest(f"swanlab local mode unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            logdir = Path(tmpdir) / "swanlab"
            run_dir = logdir / "run-20260605_030746-a3b1799d"
            run_dir.mkdir(parents=True)
            original_randint = swanlab_run_main.random.randint
            original_datetime = swankit_settings.datetime
            original_mkdir = swankit_settings.os.mkdir

            with swanlab_local_resume_patch(logdir, "a3b1799d", run_dir):
                self.assertIs(swanlab_run_main.random.randint, original_randint)
                self.assertIs(swankit_settings.datetime, original_datetime)
                self.assertIs(swankit_settings.os.mkdir, original_mkdir)

            self.assertIs(swanlab_run_main.random.randint, original_randint)
            self.assertIs(swankit_settings.datetime, original_datetime)
            self.assertIs(swankit_settings.os.mkdir, original_mkdir)


if __name__ == "__main__":
    unittest.main()
