from __future__ import annotations

import importlib
import importlib.util
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any


def import_swanlab() -> Any:
    try:
        return importlib.import_module("swanlab")
    except ImportError as exc:
        raise ImportError("swanlab is not installed, but SwanLab logging is enabled.") from exc


def swanlab_state_path(output_dir: Path) -> Path:
    return output_dir / "swanlab_run.json"


def read_swanlab_state(state_path: Path, logger: Any | None = None) -> dict[str, Any]:
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        if logger is not None:
            logger.warning("Ignoring malformed SwanLab state file: %s", state_path)
        return {}
    if not isinstance(state, dict):
        if logger is not None:
            logger.warning("Ignoring non-object SwanLab state file: %s", state_path)
        return {}
    return state


def build_swanlab_experiment_name(output_dir: Path, experiment_name: str, job_type: str) -> str:
    if experiment_name:
        return experiment_name
    if output_dir.parent.name and output_dir.parent.name != output_dir.anchor:
        return f"{job_type}_{output_dir.parent.name}_{output_dir.name}"
    return f"{job_type}_{output_dir.name}"


def find_swanlab_local_run_dir(logdir: Path, run_id: str, state: dict[str, Any]) -> Path | None:
    saved = str(state.get("local_run_dir", "")).strip()
    if saved and Path(saved).is_dir():
        return Path(saved)
    if not run_id:
        return None
    matches = sorted(logdir.glob(f"run-*-{run_id}"))
    return matches[-1] if matches else None


@contextmanager
def swanlab_local_resume_patch(logdir: Path, run_id: str, run_dir: Path | None):
    if not run_id or run_dir is None:
        yield
        return
    try:
        import swanlab.data.callbacker.local as swanlab_local
        import swanlab.data.run.main as swanlab_run_main
        import swanlab.data.sdk as swanlab_sdk
        import swankit.core.settings as swankit_settings
    except Exception:
        yield
        return

    parts = run_dir.name.split("-")
    if len(parts) < 3:
        yield
        return
    try:
        fixed_now = datetime.strptime(parts[1], "%Y%m%d_%H%M%S")
    except ValueError:
        yield
        return

    original_generate_run_id = getattr(getattr(swanlab_local, "N", None), "generate_run_id", None)
    if original_generate_run_id is None:
        yield
        return

    fixed_run_id_int = int(run_id, 16)
    original_randint = swanlab_run_main.random.randint
    original_datetime = getattr(swanlab_sdk, "datetime", None)
    original_settings_datetime = swankit_settings.datetime
    original_mkdir = getattr(getattr(swanlab_sdk, "os", None), "mkdir", None)
    original_settings_mkdir = swankit_settings.os.mkdir
    target_run_dir = run_dir.resolve()

    class FixedDatetime:
        @classmethod
        def now(cls):
            return fixed_now

    def mkdir_existing_run(path, mode=0o777, *args, **kwargs):
        if Path(path).resolve() == target_run_dir and target_run_dir.is_dir():
            return None
        return original_mkdir(path, mode, *args, **kwargs)

    def mkdir_existing_settings_run(path, mode=0o777, *args, **kwargs):
        if Path(path).resolve() == target_run_dir and target_run_dir.is_dir():
            return None
        return original_settings_mkdir(path, mode, *args, **kwargs)

    if original_generate_run_id is not None:
        swanlab_local.N.generate_run_id = lambda: run_id
    swanlab_run_main.random.randint = lambda _a, _b: fixed_run_id_int
    if original_datetime is not None:
        swanlab_sdk.datetime = FixedDatetime
    swankit_settings.datetime = FixedDatetime
    if original_mkdir is not None:
        swanlab_sdk.os.mkdir = mkdir_existing_run
    swankit_settings.os.mkdir = mkdir_existing_settings_run
    try:
        yield
    finally:
        if original_generate_run_id is not None:
            swanlab_local.N.generate_run_id = original_generate_run_id
        swanlab_run_main.random.randint = original_randint
        if original_datetime is not None:
            swanlab_sdk.datetime = original_datetime
        swankit_settings.datetime = original_settings_datetime
        if original_mkdir is not None:
            swanlab_sdk.os.mkdir = original_mkdir
        swankit_settings.os.mkdir = original_settings_mkdir


def init_swanlab_run(
    *,
    output_dir: Path,
    enabled: bool,
    mode: str,
    project: str,
    workspace: str,
    experiment_name: str,
    job_type: str,
    tags: list[str],
    config: dict[str, Any],
    logdir: str = "",
    require_swanboard_for_local: bool = False,
    logger: Any | None = None,
) -> Any | None:
    if not enabled or mode == "disabled":
        return None
    if require_swanboard_for_local and mode == "local" and importlib.util.find_spec("swanboard") is None:
        if logger is not None:
            logger.warning("Disabled SwanLab local mode because swanboard is not installed.")
        return None

    swanlab_module = import_swanlab()
    state_path = swanlab_state_path(output_dir)
    state = read_swanlab_state(state_path, logger=logger)
    run_id = str(state.get("run_id", "")).strip()

    resolved_logdir = Path(logdir) if logdir else output_dir / "swanlab"
    resolved_logdir.mkdir(parents=True, exist_ok=True)
    resolved_experiment_name = build_swanlab_experiment_name(output_dir, experiment_name, job_type)
    workspace = workspace.strip()

    init_kwargs: dict[str, Any] = {
        "project": project,
        "experiment_name": resolved_experiment_name,
        "job_type": job_type,
        "tags": list(tags),
        "config": config,
        "logdir": str(resolved_logdir),
        "mode": mode,
        "reinit": True,
    }
    if workspace:
        init_kwargs["workspace"] = workspace
    if run_id and mode == "cloud":
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    local_run_dir = find_swanlab_local_run_dir(resolved_logdir, run_id, state) if mode == "local" else None
    with swanlab_local_resume_patch(resolved_logdir, run_id, local_run_dir):
        try:
            run = swanlab_module.init(**init_kwargs)
        except Exception:
            if logger is not None:
                logger.exception("SwanLab init failed mode=%s logdir=%s", mode, resolved_logdir)
            raise

    run_public = getattr(run, "public", None)
    current_run_id = str(getattr(run_public, "run_id", "")).strip()
    current_run_dir = str(getattr(run_public, "run_dir", "")).strip()
    state_payload = {
        "project": project,
        "workspace": workspace,
        "experiment_name": resolved_experiment_name,
        "mode": mode,
        "logdir": str(resolved_logdir),
    }
    if current_run_id:
        state_payload["run_id"] = current_run_id
    if mode == "local" and current_run_dir:
        state_payload["local_run_dir"] = current_run_dir
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run
