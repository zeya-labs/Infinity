#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


KEY_METRICS = ("angle_mean", "angle_median", "angle_11_25", "angle_22_5", "angle_30")


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_dataset(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_path = dataset_dir / "metrics_summary.json"
    if not metrics_path.is_file():
        return {}
    metrics = load_json(metrics_path)
    if not isinstance(metrics, dict):
        return {}
    timing_path = dataset_dir / "inference_time_summary.json"
    timings: dict[str, Any] = {}
    if timing_path.is_file():
        payload = load_json(timing_path)
        if isinstance(payload, dict) and isinstance(payload.get("methods"), dict):
            timings = payload["methods"]
    rows: dict[str, dict[str, Any]] = {}
    for method, method_metrics in metrics.items():
        if not isinstance(method_metrics, dict):
            continue
        row = {key: value for key, value in method_metrics.items() if finite_float(value) is not None}
        method_timing = timings.get(method)
        if isinstance(method_timing, dict):
            for key in ("mean_seconds", "median_seconds", "images_per_second"):
                value = finite_float(method_timing.get(key))
                if value is not None:
                    row[f"inference_{key}"] = value
        rows[str(method)] = row
    return rows


def sort_rows(row: dict[str, Any]) -> tuple[str, float, str]:
    dataset = str(row["dataset"])
    mean = finite_float(row.get("angle_mean"))
    method = str(row["method"])
    return dataset, mean if mean is not None else float("inf"), method


def write_csv(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    columns = ["dataset", "method", *metric_keys]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def format_value(value: Any) -> str:
    parsed = finite_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.4g}"


def write_markdown(path: Path, rows: list[dict[str, Any]], metric_keys: list[str]) -> None:
    columns = ["dataset", "method", *metric_keys]
    lines = [
        "# Normal Eval Multi-Dataset Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get("dataset", "")), str(row.get("method", ""))]
        values.extend(format_value(row.get(key)) for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge per-dataset normal eval summaries.")
    parser.add_argument("--root", type=Path, required=True, help="Parent output dir containing one subdir per dataset.")
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()

    root = args.root
    datasets = [dataset.strip() for dataset in args.datasets if dataset.strip()]
    by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []
    metric_keys: set[str] = set(KEY_METRICS)
    for dataset in datasets:
        rows = collect_dataset(root / dataset)
        by_dataset[dataset] = rows
        for method, metrics in rows.items():
            row = {"dataset": dataset, "method": method, **metrics}
            flat_rows.append(row)
            metric_keys.update(str(key) for key in metrics)
    ordered_metric_keys = [key for key in KEY_METRICS if key in metric_keys]
    ordered_metric_keys.extend(sorted(metric_keys.difference(ordered_metric_keys)))
    flat_rows.sort(key=sort_rows)
    payload = {
        "root": str(root),
        "datasets": datasets,
        "rows": flat_rows,
        "by_dataset": by_dataset,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "multi_dataset_metrics_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(root / "multi_dataset_metrics_summary.csv", flat_rows, ordered_metric_keys)
    write_markdown(root / "multi_dataset_metrics_summary.md", flat_rows, ordered_metric_keys)
    print(json.dumps({"summary": str(root / "multi_dataset_metrics_summary.json"), "rows": len(flat_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
