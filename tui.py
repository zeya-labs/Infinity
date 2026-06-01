#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import re
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import DataTable, Input, Label, ListItem, ListView, Static


ROOT = Path(__file__).resolve().parent
PYTHON = str(ROOT / ".venv" / "bin" / "python") if (ROOT / ".venv" / "bin" / "python").exists() else sys.executable
TORCHRUN = str(ROOT / ".venv" / "bin" / "torchrun") if (ROOT / ".venv" / "bin" / "torchrun").exists() else "torchrun"
JOB_DIR = ROOT / ".tui" / "jobs"
VOLC_CONF_DIR = ROOT / ".tui" / "volc"
VOLC = str(Path.home() / ".volc" / "bin" / "volc") if (Path.home() / ".volc" / "bin" / "volc").exists() else "volc"
VOLC_DEFAULT_QUEUE_NAME = os.environ.get("INFINITY_VOLC_QUEUE_NAME", "queue010")
VOLC_DEFAULT_QUEUE_ID = os.environ.get("INFINITY_VOLC_QUEUE_ID", "")
VOLC_DEFAULT_IMAGE = os.environ.get(
    "INFINITY_VOLC_IMAGE",
    "cr-mlp-cn-beijing.cr.volces.com/public/cmh_test:1.6",
)
VOLC_DEFAULT_FLAVOR = os.environ.get("INFINITY_VOLC_FLAVOR", "ml.pni2.28xlarge")
VOLC_DEFAULT_GPUS = os.environ.get("INFINITY_VOLC_GPUS", "8")
VOLC_DEFAULT_FRAMEWORK = os.environ.get("INFINITY_VOLC_FRAMEWORK", "Custom")
VOLC_DEFAULT_REMOTE_ROOT = os.environ.get("INFINITY_VOLC_REMOTE_ROOT", str(ROOT))
VOLC_LOCAL_VEPFS_ROOT = Path(os.environ.get("INFINITY_VOLC_LOCAL_VEPFS_ROOT", "/root/vepfs"))
VOLC_DEFAULT_VEPFS_MOUNT = os.environ.get("INFINITY_VOLC_VEPFS_MOUNT", str(VOLC_LOCAL_VEPFS_ROOT))
VOLC_DEFAULT_ACTIVE_DEADLINE = os.environ.get("INFINITY_VOLC_ACTIVE_DEADLINE_SECONDS", "432000")
VOLC_DEFAULT_PREEMPTIBLE = os.environ.get("INFINITY_VOLC_PREEMPTIBLE", "true")
VOLC_DEFAULT_USER_CODE_PATH = os.environ.get("INFINITY_VOLC_USER_CODE_PATH", "")
VOLC_DEFAULT_REMOTE_CODE_PATH = os.environ.get("INFINITY_VOLC_REMOTE_CODE_PATH", "")
VOLC_DEFAULT_RESOURCE_FAMILY = os.environ.get("INFINITY_VOLC_RESOURCE_FAMILY", "ml.pni2")
VOLC_DEFAULT_RESOURCE_CPU = os.environ.get("INFINITY_VOLC_RESOURCE_CPU", "112")
VOLC_DEFAULT_RESOURCE_MEMORY = os.environ.get("INFINITY_VOLC_RESOURCE_MEMORY", "1960")

@dataclass
class Field:
    key: str
    label: str
    default: str
    help: str = ""
    choices: tuple[str, ...] = ()


@dataclass
class Task:
    title: str
    desc: str
    fields: list[Field]
    build: Callable[[dict[str, str]], list[str]]
    env: Callable[[dict[str, str]], dict[str, str]] = lambda values: {}
    confirm: str = ""
    category: str = "Run"
    output_slug: str = ""


@dataclass
class VolcConfig:
    queue_name: str
    queue_id: str
    image: str
    flavor: str
    gpus: str
    framework: str
    remote_root: str
    vepfs_mount: str
    active_deadline_seconds: str
    preemptible: str
    user_code_path: str
    remote_code_path: str
    resource_family: str
    resource_cpu: str
    resource_memory: str


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def shell_export(env: dict[str, str]) -> str:
    keys = ["PYTHONPATH", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES"]
    return " ".join(f"{key}={shlex.quote(env[key])}" for key in keys if env.get(key))


def pretty_command(cmd: list[str]) -> str:
    if len(cmd) <= 3:
        return shell_join(cmd)
    return " \\\n  ".join(shlex.quote(part) for part in cmd)


def tmux_safe_name(title: str, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", title).strip("_").lower()
    return f"infinity_{index + 1:02d}_{slug or 'task'}"


def tmux_managed_sessions() -> list[str]:
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#S"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return []
    return sorted(name for name in result.stdout.splitlines() if name.startswith("infinity_"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_meta_path(session_name: str) -> Path:
    return JOB_DIR / f"{session_name}.json"


def task_title_from_session(session_name: str) -> str:
    match = re.match(r"infinity_(\d+)_", session_name)
    if not match:
        return "未知任务"
    index = int(match.group(1)) - 1
    if 0 <= index < len(TASKS):
        return TASKS[index].title
    return "未知任务"


def write_job_meta(session_name: str, data: dict[str, object]) -> None:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = job_meta_path(session_name)
    existing: dict[str, object] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(data)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n")


def read_job_records() -> list[dict[str, object]]:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, object]] = {}
    live_sessions = set(tmux_managed_sessions())
    for pattern in ("infinity_*.json", "volc_*.json"):
        for path in JOB_DIR.glob(pattern):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            session_name = str(data.get("session") or path.stem)
            data["session"] = session_name
            records[session_name] = data
    for session_name in live_sessions:
        records.setdefault(
            session_name,
            {
                "session": session_name,
                "task": task_title_from_session(session_name),
                "status": "running",
                "started_at": "",
                "exit_code": "",
            },
        )
    for session_name, data in records.items():
        alive = session_name in live_sessions
        data["alive"] = alive
        if alive and data.get("status") == "running":
            data["display_status"] = "运行中"
        elif data.get("status") == "completed":
            data["display_status"] = "已完成"
        elif data.get("status") == "error":
            data["display_status"] = "报错退出"
        elif data.get("status") == "stopped":
            data["display_status"] = "已停止"
        elif data.get("status") == "submitted":
            data["display_status"] = "已提交 Volc"
        elif data.get("status") == "submit_error":
            data["display_status"] = "Volc 提交失败"
        elif data.get("status") == "running":
            data["display_status"] = "已断开"
        else:
            data["display_status"] = str(data.get("status") or "未知")
    return sorted(records.values(), key=lambda item: str(item.get("started_at") or ""), reverse=True)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return slug or "experiment"


def experiment_slug(task: Task) -> str:
    return task.output_slug or slugify(task.title)


def create_run_dir(task: Task, started_at: datetime) -> Path:
    root = ROOT / "outputs" / experiment_slug(task)
    run_dir = root / started_at.strftime("%Y-%m-%d") / started_at.strftime("%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    latest = root / "latest"
    tmp_latest = root / ".latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    tmp_latest.symlink_to(run_dir, target_is_directory=True)
    tmp_latest.replace(latest)
    return run_dir


def apply_run_outputs(values: dict[str, str], run_dir: Path) -> None:
    if "output_dir" in values:
        values["output_dir"] = str(run_dir)
    if "save_file" in values:
        suffix = Path(values["save_file"]).suffix or ".png"
        values["save_file"] = str(run_dir / f"output{suffix}")


def visible_gpu_count(cuda_devices: str) -> int | None:
    value = cuda_devices.strip()
    if not value:
        return None
    if value == "-1":
        return 0
    return len([device for device in value.split(",") if device.strip()])


def common_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if extra:
        env.update({key: value for key, value in extra.items() if value != ""})
    return env


def volc_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    volc_bin = str(Path(VOLC).parent) if Path(VOLC).exists() else str(Path.home() / ".volc" / "bin")
    env["PATH"] = f"{volc_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def current_volc_config() -> VolcConfig:
    return VolcConfig(
        queue_name=os.environ.get("INFINITY_VOLC_QUEUE_NAME", VOLC_DEFAULT_QUEUE_NAME),
        queue_id=os.environ.get("INFINITY_VOLC_QUEUE_ID", VOLC_DEFAULT_QUEUE_ID),
        image=os.environ.get("INFINITY_VOLC_IMAGE", VOLC_DEFAULT_IMAGE),
        flavor=os.environ.get("INFINITY_VOLC_FLAVOR", VOLC_DEFAULT_FLAVOR),
        gpus=os.environ.get("INFINITY_VOLC_GPUS", VOLC_DEFAULT_GPUS),
        framework=os.environ.get("INFINITY_VOLC_FRAMEWORK", VOLC_DEFAULT_FRAMEWORK),
        remote_root=os.environ.get("INFINITY_VOLC_REMOTE_ROOT", VOLC_DEFAULT_REMOTE_ROOT),
        vepfs_mount=os.environ.get("INFINITY_VOLC_VEPFS_MOUNT", VOLC_DEFAULT_VEPFS_MOUNT),
        active_deadline_seconds=os.environ.get(
            "INFINITY_VOLC_ACTIVE_DEADLINE_SECONDS",
            VOLC_DEFAULT_ACTIVE_DEADLINE,
        ),
        preemptible=os.environ.get("INFINITY_VOLC_PREEMPTIBLE", VOLC_DEFAULT_PREEMPTIBLE),
        user_code_path=os.environ.get("INFINITY_VOLC_USER_CODE_PATH", VOLC_DEFAULT_USER_CODE_PATH),
        remote_code_path=os.environ.get("INFINITY_VOLC_REMOTE_CODE_PATH", VOLC_DEFAULT_REMOTE_CODE_PATH),
        resource_family=os.environ.get("INFINITY_VOLC_RESOURCE_FAMILY", VOLC_DEFAULT_RESOURCE_FAMILY),
        resource_cpu=os.environ.get("INFINITY_VOLC_RESOURCE_CPU", VOLC_DEFAULT_RESOURCE_CPU),
        resource_memory=os.environ.get("INFINITY_VOLC_RESOURCE_MEMORY", VOLC_DEFAULT_RESOURCE_MEMORY),
    )


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def yaml_lines(value: object, indent: int = 0) -> list[str]:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                if item:
                    lines.append(f"{pad}{key}:")
                    lines.extend(yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {{}}")
            elif isinstance(item, list):
                if item:
                    lines.append(f"{pad}{key}:")
                    lines.extend(yaml_lines(item, indent + 2))
                else:
                    lines.append(f"{pad}{key}: []")
            elif isinstance(item, str) and "\n" in item:
                lines.append(f"{pad}{key}: |")
                for line in item.splitlines():
                    lines.append(f"{pad}  {line}")
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return lines
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.extend(yaml_lines(item, indent + 2))
            elif isinstance(item, str) and "\n" in item:
                lines.append(f"{pad}- |")
                for line in item.splitlines():
                    lines.append(f"{pad}  {line}")
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return lines
    return [f"{pad}{yaml_scalar(value)}"]


def write_yaml(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(yaml_lines(data)) + "\n", encoding="utf-8")


def remove_empty_config(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := remove_empty_config(item)) not in ("", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := remove_empty_config(item)) not in ("", [], {})]
    return value


def parse_positive_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} 不是合法整数: {value}") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} 必须大于 0: {value}")
    return parsed


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def cuda_list(gpu_count: int) -> str:
    return ",".join(str(index) for index in range(gpu_count))


def remoteize_value(value: str, remote_root: str) -> str:
    local_root = str(ROOT)
    normalized_remote = remote_root.rstrip("/")
    if value == local_root:
        return normalized_remote
    return value.replace(local_root + os.sep, normalized_remote + "/")


def remoteize_command(command: list[str], remote_root: str) -> list[str]:
    remote = [remoteize_value(part, remote_root) for part in command]
    if os.environ.get("INFINITY_VOLC_PYTHON") and command and command[0] == PYTHON:
        remote[0] = os.environ["INFINITY_VOLC_PYTHON"]
    if os.environ.get("INFINITY_VOLC_TORCHRUN") and command and command[0] == TORCHRUN:
        remote[0] = os.environ["INFINITY_VOLC_TORCHRUN"]
    return remote


def volc_task_name(task: Task, started_at: datetime) -> str:
    slug = re.sub(r"[^a-z0-9-]+", "-", experiment_slug(task).lower()).strip("-")
    stamp = started_at.strftime("%Y%m%d%H%M%S")
    return f"infinity-{slug or 'task'}-{stamp}"[:200]


def local_vepfs_mount(mount_path: Path) -> tuple[str, str] | None:
    try:
        result = subprocess.run(
            ["findmnt", "-T", str(mount_path), "-o", "SOURCE,TARGET", "-n"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    match = re.match(r"(?P<fs>[^\[]+)\[(?P<subpath>[^\]]*)\]\s+(?P<target>\S+)", line)
    if not match:
        return None
    try:
        relative = mount_path.relative_to(Path(match.group("target"))).as_posix()
    except ValueError:
        relative = ""
    if relative == ".":
        relative = ""
    base_subpath = match.group("subpath").rstrip("/")
    subpath = f"{base_subpath}/{relative}" if relative else base_subpath
    fs_name = match.group("fs")
    vepfs_id = fs_name.removeprefix("fs_")
    return vepfs_id, subpath.lstrip("/") or "/"


def volc_storages(config: VolcConfig) -> list[dict[str, object]]:
    raw = os.environ.get("INFINITY_VOLC_STORAGES_JSON")
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("INFINITY_VOLC_STORAGES_JSON 必须是 JSON 数组")
        return parsed
    if not config.vepfs_mount:
        return []
    storage: dict[str, object] = {
        "Type": "Vepfs",
        "MountPath": config.vepfs_mount,
        "ReadOnly": parse_bool(os.environ.get("INFINITY_VOLC_VEPFS_READ_ONLY", "false")),
    }
    mounted_vepfs = local_vepfs_mount(Path(config.vepfs_mount))
    if mounted_vepfs:
        vepfs_id, subpath = mounted_vepfs
        storage["VepfsId"] = vepfs_id
        storage["SubPath"] = subpath
    for env_key, config_key in (
        ("INFINITY_VOLC_VEPFS_ID", "VepfsId"),
        ("INFINITY_VOLC_VEPFS_NAME", "VepfsName"),
        ("INFINITY_VOLC_VEPFS_HOST_PATH", "SubPath"),
    ):
        value = os.environ.get(env_key)
        if value:
            storage[config_key] = value
    return [storage]


def volc_envs(
    config: VolcConfig,
    gpu_count: int,
    extra: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    envs = {
        "PYTHONPATH": config.remote_root,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": cuda_list(gpu_count),
    }
    if extra:
        envs.update({str(key): remoteize_value(str(value), config.remote_root) for key, value in extra.items()})
    raw = os.environ.get("INFINITY_VOLC_ENVS_JSON")
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("INFINITY_VOLC_ENVS_JSON 必须是 JSON 对象")
        envs.update({str(key): str(value) for key, value in parsed.items()})
    return [{"Name": key, "Value": value} for key, value in envs.items() if value != ""]


def volc_resource_spec(config: VolcConfig, gpu_count: int) -> dict[str, object]:
    spec: dict[str, object] = {
        "RoleName": "worker",
        "RoleReplicas": parse_positive_int(os.environ.get("INFINITY_VOLC_ROLE_REPLICAS", "1"), "Worker 数"),
    }
    if config.flavor and config.flavor != "custom":
        spec["Flavor"] = config.flavor
        return spec
    spec["Flavor"] = "custom"
    spec["ResourceSpec"] = {
        "Family": config.resource_family,
        "CPU": parse_positive_int(config.resource_cpu, "Volc CPU"),
        "Memory": parse_positive_int(config.resource_memory, "Volc Memory"),
        "GPUNum": gpu_count,
    }
    return spec


def volc_runtime_values(task: Task, values: dict[str, str], gpu_count: int) -> dict[str, str]:
    runtime = dict(values)
    if "gpus" in runtime:
        runtime["gpus"] = str(gpu_count)
    if "cuda" in runtime:
        runtime["cuda"] = cuda_list(gpu_count)
    return runtime


def build_volc_entrypoint(task: Task, values: dict[str, str], config: VolcConfig, gpu_count: int) -> tuple[str, str]:
    command = remoteize_command(task.build(values), config.remote_root)
    command_line = shell_join(command)
    env_exports = volc_envs(config, gpu_count, task.env(values))
    lines = [
        "set -e",
        f"cd {shlex.quote(config.remote_root)}",
    ]
    for env in env_exports:
        lines.append(f"export {env['Name']}={shlex.quote(env['Value'])}")
    lines += [
        f"echo {shlex.quote('$ ' + command_line)}",
        command_line,
    ]
    return "\n".join(lines), command_line


def build_volc_task_config(
    task: Task,
    values: dict[str, str],
    started_at: datetime,
    config: VolcConfig,
) -> tuple[dict[str, object], str]:
    gpu_count = parse_positive_int(config.gpus, "Volc GPU 数")
    active_deadline = parse_positive_int(config.active_deadline_seconds, "Volc 最长运行秒")
    runtime_values = volc_runtime_values(task, values, gpu_count)
    entrypoint, command_line = build_volc_entrypoint(task, runtime_values, config, gpu_count)
    task_config = remove_empty_config({
        "TaskName": volc_task_name(task, started_at),
        "Description": f"Submitted from Infinity TUI at {started_at.isoformat(timespec='seconds')}",
        "ImageUrl": config.image,
        "Framework": config.framework,
        "Entrypoint": entrypoint,
        "UserCodePath": config.user_code_path,
        "RemoteMountCodePath": config.remote_code_path,
        "TaskRoleSpecs": [volc_resource_spec(config, gpu_count)],
        "Envs": volc_envs(config, gpu_count, task.env(runtime_values)),
        "Storages": volc_storages(config),
        "ActiveDeadlineSeconds": active_deadline,
        "EnableTensorBoard": False,
        "Preemptible": parse_bool(config.preemptible),
        "Tags": ["infinity", experiment_slug(task)],
    })
    if config.queue_id:
        task_config["ResourceQueueID"] = config.queue_id
    else:
        task_config["ResourceQueueName"] = config.queue_name
    return task_config, command_line


def extract_volc_task_id(output: str) -> str:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("Id"):
        return str(parsed["Id"])
    patterns = (
        r'"Id"\s*:\s*"([A-Za-z0-9_-]+)"',
        r"task_id\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"TaskID\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"TaskId\s*[:=]\s*([A-Za-z0-9_-]+)",
        r"TaskId\s+([A-Za-z0-9_-]+)",
        r"任务ID\s*[:：]\s*([A-Za-z0-9_-]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def build_infer_2b(values: dict[str, str]) -> list[str]:
    return [
        PYTHON,
        "tools/run_infinity.py",
        "--cfg",
        values["cfg"],
        "--tau",
        values["tau"],
        "--pn",
        values["pn"],
        "--model_path",
        values["model_path"],
        "--vae_type",
        "32",
        "--vae_path",
        values["vae_path"],
        "--add_lvl_embeding_only_first_block",
        "1",
        "--use_bit_label",
        "1",
        "--model_type",
        "infinity_2b",
        "--rope2d_each_sa_layer",
        "1",
        "--rope2d_normalized_by_hw",
        "2",
        "--use_scale_schedule_embedding",
        "0",
        "--checkpoint_type",
        "torch",
        "--text_encoder_ckpt",
        values["text_encoder"],
        "--text_channels",
        "2048",
        "--apply_spatial_patchify",
        "0",
        "--prompt",
        values["prompt"],
        "--seed",
        values["seed"],
        "--save_file",
        values["save_file"],
    ]


def build_infer_8b(values: dict[str, str]) -> list[str]:
    return [
        PYTHON,
        "tools/run_infinity.py",
        "--pn",
        values["pn"],
        "--model_type",
        "infinity_8b",
        "--checkpoint_type",
        "torch_shard",
        "--model_path",
        values["model_path"],
        "--vae_type",
        "14",
        "--vae_path",
        values["vae_path"],
        "--text_encoder_ckpt",
        values["text_encoder"],
        "--text_channels",
        "2048",
        "--cfg",
        values["cfg"],
        "--tau",
        values["tau"],
        "--use_bit_label",
        "1",
        "--add_lvl_embeding_only_first_block",
        "1",
        "--rope2d_each_sa_layer",
        "1",
        "--rope2d_normalized_by_hw",
        "2",
        "--use_scale_schedule_embedding",
        "0",
        "--apply_spatial_patchify",
        "1",
        "--sampling_per_bits",
        "1",
        "--use_flex_attn",
        "0",
        "--bf16",
        "1",
        "--seed",
        values["seed"],
        "--prompt",
        values["prompt"],
        "--save_file",
        values["save_file"],
    ]


def build_normal_infer(values: dict[str, str]) -> list[str]:
    cmd = [
        PYTHON,
        "tools/run_normal_estimation.py",
        "--model-path",
        values["model_path"],
        "--input-path",
        values["input_path"],
        "--output-dir",
        values["output_dir"],
        "--seed",
        values["seed"],
        "--tau",
        values["tau"],
        "--top-k",
        values["top_k"],
        "--top-p",
        values["top_p"],
    ]
    if values["save_npy"].lower() in {"1", "yes", "true", "y"}:
        cmd.append("--save-npy")
    return cmd


def build_train_normal(values: dict[str, str]) -> list[str]:
    cmd = [
        TORCHRUN,
        "--standalone",
        "--nproc_per_node",
        values["gpus"],
        "tools/train_normal_estimation.py",
        "--output-dir",
        values["output_dir"],
        "--data-root",
        values["data_root"],
        "--normal-vae-ckpt",
        values["normal_vae"],
        "--rgb-vae-ckpt",
        values["rgb_vae"],
        "--normal-vae-type",
        values["vae_type"],
        "--rgb-vae-type",
        values["vae_type"],
        "--model-name",
        values["model_name"],
        "--init-model",
        values["init_model"],
        "--pn",
        values["pn"],
        "--batch-size",
        values["batch_size"],
        "--val-batch-size",
        values["val_batch_size"],
        "--num-workers",
        values["num_workers"],
        "--prefetch-factor",
        values["prefetch_factor"],
        "--lr",
        values["lr"],
        "--min-lr",
        values["min_lr"],
        "--warmup-ratio",
        values["warmup_ratio"],
        "--betas",
        values["beta1"],
        values["beta2"],
        "--weight-decay",
        values["weight_decay"],
        "--grad-clip",
        values["grad_clip"],
        "--precision",
        values["precision"],
        "--zero",
        values["zero"],
        "--epochs",
        values["epochs"],
        "--max-steps",
        values["max_steps"],
        "--log-every",
        values["log_every"],
        "--image-log-every",
        values["image_log_every"],
        "--ar-eval-every",
        values["ar_eval_every"],
        "--ar-eval-samples",
        values["ar_eval_samples"],
        "--ar-eval-top-k",
        values["ar_eval_top_k"],
        "--ar-eval-top-p",
        values["ar_eval_top_p"],
        "--ar-eval-tau",
        values["ar_eval_tau"],
        "--save-every-epoch",
        values["save_every_epoch"],
        "--train-partition",
        values["train_partition"],
        "--val-partition",
        values["val_partition"],
        "--max-train-samples",
        values["max_train_samples"],
        "--max-val-samples",
        values["max_val_samples"],
        "--ce-weight",
        values["ce_weight"],
        "--normal-l1-weight",
        values["normal_l1_weight"],
        "--normal-angular-weight",
        values["normal_angular_weight"],
        "--normal-latent-weight",
        values["normal_latent_weight"],
        "--normal-norm-weight",
        values["normal_norm_weight"],
        "--noise-apply-layers",
        values["noise_apply_layers"],
        "--noise-apply-strength",
        values["noise_apply_strength"],
        "--swanlab-mode",
        values["swanlab_mode"],
        "--swanlab-project",
        values["swanlab_project"],
    ]
    if values["swanlab_experiment"]:
        cmd += ["--swanlab-experiment-name", values["swanlab_experiment"]]
    if values["save_optimizer_state"].lower() in {"1", "yes", "true", "y"}:
        cmd.append("--save-optimizer-state")
    if values["spatial_patchify"].lower() in {"1", "yes", "true", "y"}:
        cmd += ["--normal-apply-spatial-patchify", "--rgb-apply-spatial-patchify"]
    else:
        cmd += ["--disable-normal-spatial-patchify", "--rgb-no-spatial-patchify"]
    if values["noise_apply_requant"].lower() in {"0", "no", "false", "n"}:
        cmd.append("--disable-noise-requant")
    return cmd


def build_train_tokenizer(values: dict[str, str]) -> list[str]:
    cmd = [
        TORCHRUN,
        "--standalone",
        "--nproc_per_node",
        values["gpus"],
        "tools/train_normal_tokenizer.py",
        "--output-dir",
        values["output_dir"],
        "--train-cache",
        values["train_cache"],
        "--val-cache",
        values["val_cache"],
        "--vae-ckpt",
        values["vae_ckpt"],
        "--batch-size",
        values["batch_size"],
        "--val-batch-size",
        values["val_batch_size"],
        "--num-workers",
        values["num_workers"],
        "--prefetch-factor",
        values["prefetch_factor"],
        "--repeat-train",
        values["repeat_train"],
        "--repeat-val",
        values["repeat_val"],
        "--lr",
        values["lr"],
        "--min-lr",
        values["min_lr"],
        "--warmup-ratio",
        values["warmup_ratio"],
        "--betas",
        values["beta1"],
        values["beta2"],
        "--weight-decay",
        values["weight_decay"],
        "--grad-clip",
        values["grad_clip"],
        "--precision",
        values["precision"],
        "--epochs",
        values["epochs"],
        "--max-steps",
        values["max_steps"],
        "--log-every",
        values["log_every"],
        "--image-log-every",
        values["image_log_every"],
        "--save-every-epoch",
        values["save_every_epoch"],
        "--recon-l1-weight",
        values["recon_l1_weight"],
        "--recon-cosine-weight",
        values["recon_cosine_weight"],
        "--vq-weight",
        values["vq_weight"],
        "--norm-weight",
        values["norm_weight"],
        "--normal-gradient-weight",
        values["normal_gradient_weight"],
        "--codebook-dim",
        values["codebook_dim"],
        "--encoder-dtype",
        values["encoder_dtype"],
        "--trainable-scope",
        values["trainable_scope"],
        "--swanlab-mode",
        values["swanlab_mode"],
        "--swanlab-project",
        values["swanlab_project"],
    ]
    if values["swanlab_experiment"]:
        cmd += ["--swanlab-experiment-name", values["swanlab_experiment"]]
    if values["mmap_cache"].lower() in {"0", "no", "false", "n"}:
        cmd.append("--disable-mmap-cache")
    if values["spatial_patchify"].lower() in {"1", "yes", "true", "y"}:
        cmd.append("--apply-spatial-patchify")
    else:
        cmd.append("--disable-spatial-patchify")
    return cmd


def build_shell(values: dict[str, str]) -> list[str]:
    return ["bash", "-lc", values["command"]]


def managed_output(prefix: str) -> str:
    return str(ROOT / "outputs" / prefix / "latest")


TASKS: list[Task] = [
    Task(
        "Infinity 2B 文生图",
        "使用本地 2B 权重生成单张图片。",
        [
            Field("prompt", "Prompt", "a cinematic portrait of a snow leopard, ultra detailed, photorealistic"),
            Field("save_file", "输出图片", "outputs/infinity_2b/latest/output.png"),
            Field("seed", "Seed", "1"),
            Field("pn", "分辨率 pn", "1M", choices=("0.06M", "0.25M", "1M")),
            Field("cfg", "CFG", "4"),
            Field("tau", "Tau", "0.5"),
            Field("model_path", "2B 权重", "weights/infinity_2b_reg.pth"),
            Field("vae_path", "VAE 权重", "weights/infinity_vae_d32_reg.pth"),
            Field("text_encoder", "文本编码器", "weights/flan-t5-xl"),
            Field("cuda", "CUDA_VISIBLE_DEVICES", "0"),
        ],
        build_infer_2b,
        env=lambda v: {"CUDA_VISIBLE_DEVICES": v["cuda"]},
        category="Inference",
        output_slug="infinity_2b",
    ),
    Task(
        "Infinity 8B 文生图",
        "使用本地 8B 权重生成单张图片。",
        [
            Field("prompt", "Prompt", "a cinematic portrait of a snow leopard wearing a tailored suit, ultra detailed"),
            Field("save_file", "输出图片", "outputs/infinity_8b/latest/output.png"),
            Field("seed", "Seed", "0"),
            Field("pn", "分辨率 pn", "1M", choices=("0.06M", "0.25M", "1M")),
            Field("cfg", "CFG", "4"),
            Field("tau", "Tau", "0.5"),
            Field("model_path", "8B 权重目录", "weights/infinity_8b_weights"),
            Field("vae_path", "VAE 权重", "weights/infinity_vae_d56_f8_14_patchify.pth"),
            Field("text_encoder", "文本编码器", "weights/flan-t5-xl"),
            Field("cuda", "CUDA_VISIBLE_DEVICES", "0"),
        ],
        build_infer_8b,
        env=lambda v: {"CUDA_VISIBLE_DEVICES": v["cuda"]},
        category="Inference",
        output_slug="infinity_8b",
    ),
    Task(
        "RGB 到 Normal 推理",
        "对单张图片或目录批量预测法线图。",
        [
            Field("input_path", "输入图片/目录", "data/infinity_toy_data/images"),
            Field("output_dir", "输出目录", managed_output("normal_infer")),
            Field("model_path", "Normal checkpoint", "outputs/normal_estimation/latest/checkpoints/last.pth"),
            Field("seed", "Seed", "0"),
            Field("tau", "Tau", "1.0"),
            Field("top_k", "Top-k", "1"),
            Field("top_p", "Top-p", "0.0"),
            Field("save_npy", "保存 npy", "0", choices=("0", "1")),
            Field("cuda", "CUDA_VISIBLE_DEVICES", "0"),
        ],
        build_normal_infer,
        env=lambda v: {"CUDA_VISIBLE_DEVICES": v["cuda"]},
        category="Normal",
        output_slug="normal_infer",
    ),
    Task(
        "训练 RGB 到 Normal",
        "启动 normal estimation 正式训练。",
        [
            Field("gpus", "GPU 数", "2"),
            Field("output_dir", "输出目录", managed_output("normal_estimation")),
            Field("data_root", "数据目录", "/root/vepfs/NormalART/datasets/processed/hypersim"),
            Field("normal_vae", "Normal VAE", "weights/infinity_vae_d32reg.pth"),
            Field("rgb_vae", "RGB VAE", "weights/infinity_vae_d32reg.pth"),
            Field("vae_type", "VAE type", "32"),
            Field("spatial_patchify", "Spatial patchify", "0", choices=("0", "1")),
            Field("model_name", "模型名", "infinity_2b"),
            Field("init_model", "初始化模型", "weights/infinity_2b_reg.pth"),
            Field("pn", "分辨率 pn", "0.06M", choices=("0.06M", "0.25M", "1M")),
            Field("batch_size", "Batch/GPU", "4"),
            Field("val_batch_size", "Val batch/GPU", "4"),
            Field("num_workers", "Workers", "4"),
            Field("prefetch_factor", "Prefetch factor", "4"),
            Field("lr", "Learning rate", "1e-4"),
            Field("min_lr", "Min LR", "1e-5"),
            Field("warmup_ratio", "Warmup ratio", "0.03"),
            Field("beta1", "Adam beta1", "0.9"),
            Field("beta2", "Adam beta2", "0.95"),
            Field("weight_decay", "Weight decay", "1e-4"),
            Field("grad_clip", "Grad clip", "1.0"),
            Field("precision", "Precision", "bf16", choices=("bf16", "fp16", "fp32")),
            Field("zero", "ZeRO", "0", choices=("0", "2", "3")),
            Field("epochs", "Epochs", "20"),
            Field("max_steps", "Max steps", "0"),
            Field("log_every", "Log every", "10"),
            Field("image_log_every", "Image log every", "200"),
            Field("ar_eval_every", "AR eval every", "0"),
            Field("ar_eval_samples", "AR eval samples", "32"),
            Field("ar_eval_top_k", "AR top-k", "1"),
            Field("ar_eval_top_p", "AR top-p", "0.0"),
            Field("ar_eval_tau", "AR tau", "1.0"),
            Field("save_every_epoch", "Save every epoch", "1"),
            Field("save_optimizer_state", "保存优化器", "0", choices=("0", "1")),
            Field("train_partition", "Train split", "train"),
            Field("val_partition", "Val split", "val"),
            Field("max_train_samples", "Max train samples", "0"),
            Field("max_val_samples", "Max val samples", "0"),
            Field("ce_weight", "CE weight", "1.0"),
            Field("normal_l1_weight", "Normal L1 weight", "0.25"),
            Field("normal_angular_weight", "Angular weight", "0.5"),
            Field("normal_latent_weight", "Latent weight", "0.1"),
            Field("normal_norm_weight", "Norm weight", "0.05"),
            Field("noise_apply_layers", "Noise layers", "-1"),
            Field("noise_apply_strength", "Noise strength", "0.0"),
            Field("noise_apply_requant", "Noise requant", "1", choices=("0", "1")),
            Field("swanlab_mode", "SwanLab", "local", choices=("local", "cloud", "offline", "disabled")),
            Field("swanlab_project", "SwanLab project", "infinity_normal_estimation_hypersim"),
            Field("swanlab_experiment", "SwanLab experiment", ""),
            Field("cuda", "CUDA_VISIBLE_DEVICES", "0,1"),
        ],
        build_train_normal,
        env=lambda v: {"CUDA_VISIBLE_DEVICES": v["cuda"]},
        confirm="训练任务会长时间占用 GPU，当前默认是正式训练参数。",
        category="Train",
        output_slug="normal_estimation",
    ),
    Task(
        "训练法线 Tokenizer",
        "启动 normal tokenizer 正式微调。",
        [
            Field("gpus", "GPU 数", "2"),
            Field("output_dir", "输出目录", managed_output("normal_tokenizer")),
            Field("train_cache", "训练 cache", "data/normalart/datasets/cache/hypersim_full/256x256/train.pt"),
            Field("val_cache", "验证 cache", "data/normalart/datasets/cache/hypersim_full/256x256/val.pt"),
            Field("vae_ckpt", "基础 VAE", "weights/infinity_vae_d32reg.pth"),
            Field("batch_size", "Batch/GPU", "16"),
            Field("val_batch_size", "Val batch/GPU", "16"),
            Field("num_workers", "Workers", "8"),
            Field("prefetch_factor", "Prefetch factor", "4"),
            Field("repeat_train", "Repeat train", "1"),
            Field("repeat_val", "Repeat val", "1"),
            Field("mmap_cache", "MMap cache", "1", choices=("0", "1")),
            Field("lr", "Learning rate", "2e-4"),
            Field("min_lr", "Min LR", "1e-5"),
            Field("warmup_ratio", "Warmup ratio", "0.03"),
            Field("beta1", "Adam beta1", "0.9"),
            Field("beta2", "Adam beta2", "0.95"),
            Field("weight_decay", "Weight decay", "1e-4"),
            Field("grad_clip", "Grad clip", "1.0"),
            Field("precision", "Precision", "bf16", choices=("bf16", "fp16", "fp32")),
            Field("epochs", "Epochs", "20"),
            Field("max_steps", "Max steps", "0"),
            Field("log_every", "Log every", "10"),
            Field("image_log_every", "Image log every", "200"),
            Field("save_every_epoch", "Save every epoch", "1"),
            Field("recon_l1_weight", "Recon L1 weight", "1.0"),
            Field("recon_cosine_weight", "Recon cosine weight", "1.0"),
            Field("vq_weight", "VQ weight", "1.0"),
            Field("norm_weight", "Norm weight", "0.1"),
            Field("normal_gradient_weight", "Normal gradient weight", "0.2"),
            Field("codebook_dim", "Codebook dim", "32"),
            Field("spatial_patchify", "Spatial patchify", "0", choices=("0", "1")),
            Field("encoder_dtype", "Encoder dtype", "bf16", choices=("bf16", "fp32")),
            Field("trainable_scope", "训练范围", "decoder_only", choices=("all", "decoder_quantizer", "decoder_only")),
            Field("swanlab_mode", "SwanLab", "local", choices=("local", "cloud", "offline", "disabled")),
            Field("swanlab_project", "SwanLab project", "infinity_normal_tokenizer_hypersim"),
            Field("swanlab_experiment", "SwanLab experiment", ""),
            Field("cuda", "CUDA_VISIBLE_DEVICES", "0,1"),
        ],
        build_train_tokenizer,
        env=lambda v: {"CUDA_VISIBLE_DEVICES": v["cuda"]},
        confirm="训练任务会长时间占用 GPU，当前默认是正式训练参数。",
        category="Train",
        output_slug="normal_tokenizer",
    ),
    Task("评估", "运行 scripts/eval.sh。请先检查脚本内变量配置。", [Field("command", "命令", "bash scripts/eval.sh")], build_shell, confirm="评估脚本可能安装/调用多个依赖。", category="Eval"),
    Task("GPU 状态", "显示 nvidia-smi。", [Field("command", "命令", "nvidia-smi")], build_shell, category="Utility"),
    Task(
        "检查权重和输出",
        "列出常用权重、latest 输出链接和 checkpoint。",
        [
            Field(
                "command",
                "命令",
                "ls -lh weights/infinity_2b_reg.pth weights/infinity_vae_d32_reg.pth weights/infinity_vae_d56_f8_14_patchify.pth 2>/dev/null; "
                "ls -lh outputs/*/latest 2>/dev/null; "
                "find outputs/normal_estimation/latest/checkpoints outputs/normal_tokenizer/latest/checkpoints -maxdepth 1 -type f 2>/dev/null | sort | tail -30",
            )
        ],
        build_shell,
        category="Utility",
    ),
    Task("自定义命令", "在项目根目录执行一条 shell 命令。", [Field("command", "命令", "echo hello")], build_shell, category="Utility"),
]


CSS = """
Screen {
    background: #1a1b26;
    color: #cfc9c2;
}

#shell {
    height: 100%;
    layout: vertical;
}

#topbar {
    height: 2;
    padding: 0 1;
    background: #1a1b26;
}

#brand {
    width: 22;
    padding: 0 1;
    content-align: left middle;
    text-style: bold;
    color: #7dcfff;
    background: #1a1b26;
}

#root_path {
    width: 1fr;
    padding: 0 1;
    content-align: left middle;
    color: #565f89;
}

#top_status {
    width: 5;
    padding: 0 1;
    content-align: right middle;
    color: #9ece6a;
    text-style: bold;
    transition: color 160ms in_out_cubic;
}

#workspace {
    height: 1fr;
}

#keybar {
    height: 1;
    background: #1a1b26;
    color: #565f89;
}

#keybar_spacer {
    width: 33;
    min-width: 30;
}

#keybar_text {
    width: 1fr;
    padding: 0 1;
    color: #565f89;
}

#sidebar {
    width: 31;
    min-width: 28;
    margin: 0 1 0 1;
    padding: 1;
    background: #1a1b26;
    border: round #565f89;
}

#nav_title {
    height: 1;
    padding: 0 1;
    content-align: left middle;
    text-style: bold;
    color: #7dcfff;
}

#task_list {
    height: 1fr;
    padding: 1 1 1 1;
}

ListItem {
    height: 1;
    padding: 0 1;
    color: #cfc9c2;
}

ListItem.-highlight {
    background: #414868;
    color: #ffffff;
    text-style: bold;
}

#main {
    width: 1fr;
    padding: 0 1 0 0;
}

#task_header {
    height: 5;
    padding: 0 2;
    background: #1a1b26;
    border: round #565f89;
}

#top_panel {
    height: 1fr;
    min-height: 18;
    margin-top: 1;
}

#details {
    width: 34%;
    min-width: 25;
    margin: 0 1 0 0;
    padding: 1;
    background: #1a1b26;
    border: round #565f89;
}

#params {
    width: 66%;
    padding: 1;
    background: #1a1b26;
    border: round #565f89;
}

.panel_label {
    height: 1;
    margin-bottom: 1;
    content-align: left middle;
    color: #7dcfff;
    text-style: bold;
}

.section_label {
    height: 1;
    margin-top: 1;
    color: #7dcfff;
    text-style: bold;
}

#task_title {
    height: 1;
    content-align: left middle;
    text-style: bold;
    color: #cfc9c2;
}

#task_badge {
    height: 1;
    color: #9ece6a;
}

#task_desc {
    height: 1;
    color: #a9b1d6;
}

#command_preview {
    height: 1fr;
    background: #16161e;
    scrollbar-size-vertical: 1;
}

#command_text {
    width: 100%;
    color: #cfc9c2;
    background: #16161e;
    padding: 1 1;
}

#field_table {
    height: 1fr;
    background: #1a1b26;
}

DataTable {
    background: #1a1b26;
    color: #cfc9c2;
    scrollbar-size-vertical: 1;
}

SessionsScreen {
    align: center middle;
}

EditFieldScreen {
    align: center middle;
}

ChoiceFieldScreen {
    align: center middle;
}

#edit_modal {
    width: 86;
    height: 9;
    padding: 1;
    background: #1a1b26;
    border: round #7dcfff;
}

#edit_title {
    height: 1;
    color: #7dcfff;
    text-style: bold;
}

#edit_input {
    height: 3;
    margin-top: 1;
    background: #16161e;
    color: #cfc9c2;
    border: tall #565f89;
}

#edit_hint {
    height: 1;
    color: #565f89;
}

#choice_modal {
    width: 64;
    height: auto;
    max-height: 18;
    padding: 1;
    background: #1a1b26;
    border: round #7dcfff;
}

#choice_title {
    height: 1;
    color: #7dcfff;
    text-style: bold;
}

#choice_hint {
    height: 1;
    color: #565f89;
}

#choice_table {
    height: auto;
    max-height: 12;
    margin-top: 1;
    background: #1a1b26;
}

#session_modal {
    width: 92;
    height: 23;
    padding: 1;
    background: #1a1b26;
    border: round #7dcfff;
}

#session_title {
    height: 1;
    color: #7dcfff;
    text-style: bold;
}

#session_hint {
    height: 1;
    color: #565f89;
}

#session_table {
    height: 1fr;
    margin-top: 1;
    background: #1a1b26;
}
"""


class SessionsScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "关闭"),
        ("q", "cancel", "关闭"),
        ("a", "attach_selected", "进入"),
        ("enter", "attach_selected", "进入"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.records = read_job_records()

    def compose(self) -> ComposeResult:
        with Vertical(id="session_modal"):
            yield Static("任务会话", id="session_title")
            yield Static("↑↓ 选择    Enter/a 进入可 attach 的 tmux    Esc/q 关闭", id="session_hint")
            table = DataTable(
                id="session_table",
                cursor_type="row",
                show_row_labels=False,
                zebra_stripes=True,
                cell_padding=1,
            )
            table.add_columns("状态", "任务", "Session", "Exit", "开始时间")
            yield table

    def on_mount(self) -> None:
        table = self.query_one("#session_table", DataTable)
        for record in self.records:
            exit_code = record.get("exit_code")
            table.add_row(
                str(record.get("display_status") or ""),
                str(record.get("task") or ""),
                str(record.get("session") or ""),
                "" if exit_code in {None, ""} else str(exit_code),
                str(record.get("started_at") or ""),
            )
        if not self.records:
            table.add_row("空", "没有 TUI 管理的任务", "", "", "")

    @on(DataTable.RowSelected, "#session_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.action_attach_selected()

    def action_attach_selected(self) -> None:
        table = self.query_one("#session_table", DataTable)
        row_index = table.cursor_row
        if row_index is None or row_index < 0 or row_index >= len(self.records):
            self.dismiss(None)
            return
        record = self.records[row_index]
        session_name = str(record.get("session") or "")
        if not session_name or not bool(record.get("alive")):
            self.notify("这个任务已经没有可 attach 的 tmux session", severity="warning")
            return
        self.dismiss(session_name)

    def action_cancel(self) -> None:
        self.dismiss(None)


class EditFieldScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "取消"),
        ("ctrl+c", "cancel", "取消"),
    ]

    def __init__(self, label: str, value: str) -> None:
        super().__init__()
        self.label = label
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="edit_modal"):
            yield Static(self.label, id="edit_title")
            yield Input(value=self.value, id="edit_input")
            yield Static("Enter 保存    Esc 取消", id="edit_hint")

    def on_mount(self) -> None:
        editor = self.query_one("#edit_input", Input)
        editor.focus()
        editor.cursor_position = len(editor.value)

    @on(Input.Submitted, "#edit_input")
    def on_editor_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceFieldScreen(ModalScreen[str | None]):
    BINDINGS = [
        ("escape", "cancel", "取消"),
        ("q", "cancel", "取消"),
        ("enter", "choose", "选择"),
    ]

    def __init__(self, label: str, value: str, choices: tuple[str, ...]) -> None:
        super().__init__()
        self.label = label
        self.value = value
        self.choices = choices

    def compose(self) -> ComposeResult:
        with Vertical(id="choice_modal"):
            yield Static(self.label, id="choice_title")
            yield Static("↑↓ 选择    Enter/点击 保存    Esc 取消", id="choice_hint")
            table = DataTable(
                id="choice_table",
                cursor_type="row",
                show_header=False,
                show_row_labels=False,
                zebra_stripes=True,
                cell_padding=1,
            )
            table.add_column("值")
            for choice in self.choices:
                marker = "●" if choice == self.value else " "
                table.add_row(f"{marker}  {choice}")
            yield table

    def on_mount(self) -> None:
        table = self.query_one("#choice_table", DataTable)
        if self.value in self.choices:
            table.move_cursor(row=self.choices.index(self.value), column=0)
        table.focus()

    @on(DataTable.RowSelected, "#choice_table")
    def on_row_selected(self, event: DataTable.RowSelected) -> None:
        self.choose_row(event.cursor_row)

    def action_choose(self) -> None:
        table = self.query_one("#choice_table", DataTable)
        self.choose_row(table.cursor_row)

    def choose_row(self, row_index: int | None) -> None:
        if row_index is None or row_index < 0 or row_index >= len(self.choices):
            return
        self.dismiss(self.choices[row_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class InfinityTUI(App):
    CSS = CSS
    BINDINGS = [
        ("q", "quit", "退出"),
        ("r", "run_task", "运行"),
        ("v", "submit_volc_task", "提交 Volc"),
        ("a", "attach_task", "进入终端"),
        ("s", "show_sessions", "会话"),
        ("c", "copy_command", "显示命令"),
        ("ctrl+c", "stop_task", "停止任务"),
    ]

    selected_index = reactive(0)

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[int, dict[str, str]] = {
            idx: {field.key: field.default for field in task.fields} for idx, task in enumerate(TASKS)
        }
        self.session_name: str | None = None
        self.live_sessions: set[str] = set()
        self.preview_cache: dict[int, str] = {}
        self.refresh_timer: Timer | None = None
        self.task_title: Static | None = None
        self.task_badge: Static | None = None
        self.task_desc: Static | None = None
        self.top_status: Static | None = None
        self.field_table: DataTable | None = None
        self.command_text: Static | None = None
        self.command_preview: VerticalScroll | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="shell"):
            with Horizontal(id="topbar"):
                yield Static("Infinity Lab", id="brand")
                yield Static(str(ROOT), id="root_path")
                yield Static("○", id="top_status")
            with Horizontal(id="workspace"):
                with Vertical(id="sidebar"):
                    yield Static("Experiments", id="nav_title")
                    yield ListView(
                        *(ListItem(Label(f"{idx + 1:02d}  {task.title}")) for idx, task in enumerate(TASKS)),
                        id="task_list",
                    )
                with Vertical(id="main"):
                    with Vertical(id="task_header"):
                        yield Static("", id="task_title")
                        yield Static("", id="task_badge")
                        yield Static("", id="task_desc")
                    with Horizontal(id="top_panel"):
                        with Vertical(id="details"):
                            yield Static("COMMAND", classes="panel_label")
                            with VerticalScroll(id="command_preview"):
                                yield Static("", id="command_text")
                        with Vertical(id="params"):
                            yield Static("PARAMETERS", classes="panel_label")
                            table = DataTable(
                                id="field_table",
                                cursor_type="row",
                                show_row_labels=False,
                                zebra_stripes=True,
                                cell_padding=1,
                            )
                            table.add_columns("参数", "值")
                            yield table
            with Horizontal(id="keybar"):
                yield Static("", id="keybar_spacer")
                yield Static(
                    "q 退出    ↑↓ 选择任务    Enter/点击参数 编辑    r 本地运行    v 提交 Volc    a 进入 tmux    s 会话    c 命令",
                    id="keybar_text",
                )

    def on_mount(self) -> None:
        self.task_title = self.query_one("#task_title", Static)
        self.task_badge = self.query_one("#task_badge", Static)
        self.task_desc = self.query_one("#task_desc", Static)
        self.top_status = self.query_one("#top_status", Static)
        self.field_table = self.query_one("#field_table", DataTable)
        self.command_text = self.query_one("#command_text", Static)
        self.command_preview = self.query_one("#command_preview", VerticalScroll)
        self.live_sessions = set(tmux_managed_sessions())
        self.query_one("#task_list", ListView).index = 0
        self.refresh_task()

    @on(ListView.Highlighted, "#task_list")
    def on_task_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "task_list" or event.item is None:
            return
        index = event.list_view.index or 0
        if index == self.selected_index:
            return
        self.selected_index = index
        self.schedule_task_refresh()

    @on(DataTable.RowSelected, "#field_table")
    def on_field_row_selected(self, event: DataTable.RowSelected) -> None:
        self.edit_field(event.cursor_row)

    @on(DataTable.CellSelected, "#field_table")
    def on_field_cell_selected(self, event: DataTable.CellSelected) -> None:
        self.edit_field(event.coordinate.row)

    def current_task(self) -> Task:
        return TASKS[self.selected_index]

    def current_values(self) -> dict[str, str]:
        return self.values[self.selected_index]

    def current_session_name(self) -> str:
        return tmux_safe_name(self.current_task().title, self.selected_index)

    def set_status_icon(self, icon: str) -> None:
        if self.top_status is not None:
            self.top_status.update(icon)

    def schedule_task_refresh(self) -> None:
        if self.refresh_timer is not None:
            self.refresh_timer.stop()
        self.refresh_timer = self.set_timer(0.035, self.flush_task_refresh)

    def flush_task_refresh(self) -> None:
        self.refresh_timer = None
        self.refresh_task()

    def refresh_task(self, keep_cursor: int | None = None) -> None:
        task = self.current_task()
        values = self.current_values()
        if self.task_title is not None:
            self.task_title.update(task.title)
        if self.task_badge is not None:
            self.task_badge.update(f"{task.category}  |  {len(task.fields)} 个参数")
        if self.task_desc is not None:
            self.task_desc.update(task.desc)
        session_name = self.current_session_name()
        self.set_status_icon("●" if session_name in self.live_sessions else "○")
        table = self.field_table
        if table is None:
            return
        table.clear()
        for field in task.fields:
            help_suffix = f"  ({field.help})" if field.help else ""
            table.add_row(f"{field.label}{help_suffix}", values[field.key])
        if keep_cursor is not None and task.fields:
            table.move_cursor(row=min(keep_cursor, len(task.fields) - 1), column=0)
        self.update_command_preview()
        if session_name in self.live_sessions:
            self.session_name = session_name

    def edit_field(self, row_index: int | None) -> None:
        task = self.current_task()
        if row_index is None or row_index < 0 or row_index >= len(task.fields):
            return
        field = task.fields[row_index]
        value = self.current_values()[field.key]
        callback = lambda new_value: self.apply_field_edit(row_index, new_value)
        if field.choices:
            self.push_screen(ChoiceFieldScreen(field.label, value, field.choices), callback)
        else:
            self.push_screen(EditFieldScreen(field.label, value), callback)

    def apply_field_edit(self, row_index: int, value: str | None) -> None:
        if value is None:
            return
        task = self.current_task()
        if row_index < 0 or row_index >= len(task.fields):
            return
        field = task.fields[row_index]
        self.values[self.selected_index][field.key] = value
        self.preview_cache.pop(self.selected_index, None)
        if self.field_table is not None:
            self.field_table.update_cell_at((row_index, 1), value, update_width=True)
        self.update_command_preview()
        self.set_status_icon("○")

    def update_command_preview(self) -> None:
        command = self.preview_cache.get(self.selected_index)
        if command is None:
            task = self.current_task()
            try:
                values = self.current_values()
                env = common_env(task.env(values))
                prefix = shell_export(env)
                command = pretty_command(task.build(values))
                if prefix:
                    command = f"{prefix} \\\n  {command}"
            except Exception as exc:
                command = f"命令生成失败: {exc}"
            self.preview_cache[self.selected_index] = command
        if self.command_text is not None:
            self.command_text.update(command)

    def action_copy_command(self) -> None:
        task = self.current_task()
        values = self.current_values()
        env = common_env(task.env(values))
        prefix = shell_export(env)
        command = shell_join(task.build(values))
        shown = f"{prefix} {command}" if prefix else command
        if self.command_text is not None:
            self.command_text.update(shown)
        if self.command_preview is not None:
            self.command_preview.scroll_home(animate=False)
        self.set_status_icon("⌘")

    def tmux_session_alive(self, session_name: str | None = None) -> bool:
        name = session_name or self.session_name
        if not name:
            return False
        if name in self.live_sessions:
            return True
        result = subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        alive = result.returncode == 0
        if alive:
            self.live_sessions.add(name)
        else:
            self.live_sessions.discard(name)
        return alive

    def resolve_session_name(self) -> str | None:
        current = self.current_session_name()
        if self.tmux_session_alive(current):
            self.session_name = current
            return current
        if self.tmux_session_alive(self.session_name):
            return self.session_name
        self.session_name = None
        return None

    def action_show_sessions(self) -> None:
        self.live_sessions = set(tmux_managed_sessions())
        self.push_screen(SessionsScreen(), self.attach_session_from_picker)
        self.set_status_icon("▤")

    def attach_session_from_picker(self, session_name: str | None) -> None:
        if not session_name:
            self.refresh_task()
            return
        self.attach_tmux_session(session_name)

    def action_stop_task(self) -> None:
        session_name = self.resolve_session_name()
        if session_name:
            write_job_meta(session_name, {"status": "stopped", "ended_at": now_utc()})
            subprocess.run(["tmux", "kill-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.live_sessions.discard(session_name)
            self.set_status_icon("■")
            if self.session_name == session_name:
                self.session_name = None
        else:
            self.notify("没有正在运行的 tmux session", severity="warning")

    def action_run_task(self) -> None:
        running_session = self.resolve_session_name()
        if running_session:
            self.set_status_icon("●")
            return
        task = self.current_task()
        session_name = self.current_session_name()
        values = dict(self.current_values())
        if values.get("gpus") and values.get("cuda"):
            try:
                requested_gpus = int(values["gpus"])
            except ValueError:
                self.notify(f"GPU 数不是合法整数: {values['gpus']}", severity="error")
                self.set_status_icon("!")
                return
            visible_gpus = visible_gpu_count(values["cuda"])
            if visible_gpus is not None and requested_gpus > visible_gpus:
                self.notify(
                    f"GPU 数={requested_gpus}，但 CUDA_VISIBLE_DEVICES 只有 {visible_gpus} 张可见卡",
                    severity="error",
                )
                self.set_status_icon("!")
                return
        started_at = datetime.now(timezone.utc)
        run_dir = create_run_dir(task, started_at)
        apply_run_outputs(values, run_dir)
        command = task.build(values)
        env = common_env(task.env(values))
        prefix = shell_export(env)
        cmd = shell_join(command)
        run_line = f"{prefix} {cmd}" if prefix else cmd
        confirm = f"echo {shlex.quote('[提示] ' + task.confirm)}; " if task.confirm else ""
        meta_path = job_meta_path(session_name)
        meta_update_code = (
            "import json, pathlib, sys, datetime; "
            "p=pathlib.Path(sys.argv[1]); "
            "data=json.loads(p.read_text()); "
            "rc=int(sys.argv[2]); "
            "data.update({'status':'completed' if rc == 0 else 'error', "
            "'exit_code':rc, "
            "'ended_at':datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}); "
            "p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\\n')"
        )
        meta_update = (
            f"{shlex.quote(PYTHON)} -c "
            f"{shlex.quote(meta_update_code)} "
            f"{shlex.quote(str(meta_path))} \"$rc\""
        )
        script = (
            f"cd {shlex.quote(str(ROOT))}; "
            f"{confirm}"
            f"echo {shlex.quote('$ ' + run_line)}; "
            f"{run_line}; "
            "rc=$?; echo; echo \"[exit code: $rc]\"; "
            f"{meta_update}; "
            "echo; echo \"任务结束。按 Ctrl-b d 返回 TUI，或输入 exit 关闭 session。\"; "
            "exec $SHELL"
        )
        write_job_meta(
            session_name,
            {
                "session": session_name,
                "task": task.title,
                "category": task.category,
                "status": "running",
                "started_at": started_at.isoformat(timespec="seconds"),
                "exit_code": "",
                "command": run_line,
                "run_dir": str(run_dir),
                "latest": str(ROOT / "outputs" / experiment_slug(task) / "latest"),
            },
        )
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", session_name, script],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            self.notify(f"tmux 启动失败: {result.stdout.strip()}", severity="error")
            self.set_status_icon("!")
            return
        self.session_name = session_name
        self.live_sessions.add(session_name)
        self.set_status_icon("●")

    def action_submit_volc_task(self) -> None:
        if not Path(VOLC).exists() and shutil.which(VOLC) is None:
            self.notify("未找到 volc CLI，请先安装并配置火山引擎命令行工具", severity="error")
            self.set_status_icon("!")
            return
        task = self.current_task()
        values = dict(self.current_values())
        started_at = datetime.now(timezone.utc)
        run_dir = create_run_dir(task, started_at)
        apply_run_outputs(values, run_dir)
        config = current_volc_config()
        try:
            task_config, command_line = build_volc_task_config(task, values, started_at, config)
        except Exception as exc:
            self.notify(f"Volc 配置生成失败: {exc}", severity="error")
            self.set_status_icon("!")
            return
        task_name = str(task_config["TaskName"])
        conf_path = VOLC_CONF_DIR / f"{task_name}.yaml"
        try:
            write_yaml(conf_path, task_config)
        except OSError as exc:
            self.notify(f"写入 Volc 配置失败: {exc}", severity="error")
            self.set_status_icon("!")
            return
        submit_cmd = [VOLC, "ml_task", "submit", "-c", str(conf_path)]
        result = subprocess.run(
            submit_cmd,
            cwd=ROOT,
            env=volc_cli_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout.strip()
        meta_name = f"volc_{task_name}"
        if result.returncode != 0:
            write_job_meta(
                meta_name,
                {
                    "session": meta_name,
                    "task": task.title,
                    "category": f"{task.category}/Volc",
                    "status": "submit_error",
                    "started_at": started_at.isoformat(timespec="seconds"),
                    "exit_code": result.returncode,
                    "command": shell_join(submit_cmd),
                    "volc_command": command_line,
                    "volc_conf": str(conf_path),
                    "volc_flavor": config.flavor,
                    "volc_gpus": config.gpus,
                    "volc_preemptible": config.preemptible,
                    "run_dir": str(run_dir),
                    "latest": str(ROOT / "outputs" / experiment_slug(task) / "latest"),
                    "submit_output": output,
                },
            )
            self.notify(f"Volc 提交失败: {output[-220:] or result.returncode}", severity="error")
            self.set_status_icon("!")
            return
        volc_task_id = extract_volc_task_id(output)
        write_job_meta(
            meta_name,
            {
                "session": meta_name,
                "task": task.title,
                "category": f"{task.category}/Volc",
                "status": "submitted",
                "started_at": started_at.isoformat(timespec="seconds"),
                "exit_code": "",
                "command": shell_join(submit_cmd),
                "volc_command": command_line,
                "volc_task_name": task_name,
                "volc_task_id": volc_task_id,
                "volc_conf": str(conf_path),
                "volc_flavor": config.flavor,
                "volc_gpus": config.gpus,
                "volc_preemptible": config.preemptible,
                "run_dir": str(run_dir),
                "latest": str(ROOT / "outputs" / experiment_slug(task) / "latest"),
                "submit_output": output,
            },
        )
        label = volc_task_id or task_name
        resource_label = f"{config.flavor}/{config.gpus}卡/闲时"
        self.notify(f"已提交 Volc: {label} ({resource_label})", severity="information")
        self.set_status_icon("◇")

    def action_attach_task(self) -> None:
        session_name = self.resolve_session_name()
        if not session_name:
            self.action_show_sessions()
            return
        self.attach_tmux_session(session_name)

    def attach_tmux_session(self, session_name: str) -> None:
        self.set_status_icon("↗")
        with self.suspend():
            subprocess.run(["tmux", "attach-session", "-t", session_name])
        if self.tmux_session_alive(session_name):
            self.session_name = session_name
            self.set_status_icon("●")
        else:
            self.set_status_icon("○")
            self.live_sessions.discard(session_name)
            self.session_name = None


def main() -> int:
    InfinityTUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
