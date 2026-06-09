#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a one-GPU Volc task for normal uncertainty inference.")
    parser.add_argument("--input-path", required=True, help="Image file or image directory visible from the VEPFS mount.")
    parser.add_argument(
        "--model-path",
        default="outputs/normal_estimation/2026-06-05/03-07-01/checkpoints/best_angle_20.7002.pth",
    )
    parser.add_argument("--output-dir", default="outputs/normal_uncertainty/volc_run")
    parser.add_argument("--pn", default="", choices=("", "0.06M", "0.25M", "1M"))
    parser.add_argument("--normal-vae-ckpt", default="")
    parser.add_argument("--normal-vae-type", default="")
    parser.add_argument("--timing-warmup", default="0")
    parser.add_argument("--timing-repeats", default="1")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    command = [
        tui.PYTHON,
        "tools/run_normal_estimation.py",
        "--model-path",
        args.model_path,
        "--input-path",
        args.input_path,
        "--output-dir",
        args.output_dir,
        "--save-uncertainty",
        "--save-npy",
        "--timing-warmup",
        args.timing_warmup,
        "--timing-repeats",
        args.timing_repeats,
    ]
    if args.pn:
        command.extend(["--pn", args.pn])
    if args.normal_vae_ckpt:
        command.extend(["--normal-vae-ckpt", args.normal_vae_ckpt])
    if args.normal_vae_type:
        command.extend(["--normal-vae-type", args.normal_vae_type])
    task = tui.Task(
        "法线不确定图",
        "Run normal estimation and save AR-token entropy uncertainty.",
        [],
        lambda _values: command,
        category="Utility",
        output_slug="normal_uncertainty",
    )
    config = tui.current_volc_config()
    config.gpus = "1"
    started_at = datetime.now(timezone.utc)
    task_config, command_line = tui.build_volc_task_config(task, {}, started_at, config)
    conf_path = tui.VOLC_CONF_DIR / f"{task_config['TaskName']}.yaml"
    tui.write_yaml(conf_path, task_config)
    print(f"Volc config: {conf_path}")
    print(f"Command: {command_line}")
    if args.dry_run:
        return 0
    submit_cmd = [tui.VOLC, "ml_task", "submit", "-c", str(conf_path), "--priority", config.priority]
    result = subprocess.run(
        submit_cmd,
        cwd=tui.ROOT,
        env=tui.volc_cli_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
