"""Collect YOLO experiment outputs into thesis-ready CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


METRIC_ALIASES = {
    "precision": ["metrics/precision(B)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall"],
    "map50": ["metrics/mAP50(B)", "metrics/mAP_0.5", "metrics/mAP50"],
    "map50_95": ["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "metrics/mAP50-95"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize YOLO results for thesis tables.")
    parser.add_argument("--runs", type=Path, default=Path("runs/remote_yolo"))
    parser.add_argument("--speed", type=Path, default=Path("reports/speed_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/thesis_results.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/thesis_results.md"))
    return parser.parse_args()


def find_metric(row: pd.Series, aliases: list[str]) -> float | None:
    for key in aliases:
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return None


def select_best_row(df: pd.DataFrame) -> pd.Series:
    for key in METRIC_ALIASES["map50_95"]:
        if key in df.columns and df[key].notna().any():
            return df.loc[df[key].idxmax()]
    for key in METRIC_ALIASES["map50"]:
        if key in df.columns and df[key].notna().any():
            return df.loc[df[key].idxmax()]
    return df.iloc[-1]


def read_args_yaml(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "args.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def checkpoint_size_mb(run_dir: Path) -> float | None:
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        return None
    return best.stat().st_size / 1024 / 1024


def load_speed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "experiment" not in df.columns:
        return {}
    return {str(row["experiment"]): row.to_dict() for _, row in df.iterrows()}


def summarize_run(run_dir: Path, speed_rows: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return None

    df = pd.read_csv(results_csv)
    if df.empty:
        return None

    row = select_best_row(df)
    args = read_args_yaml(run_dir)
    summary = {
        "experiment": run_dir.name,
        "model": args.get("model", ""),
        "imgsz": args.get("imgsz", ""),
        "epochs": int(row.get("epoch", len(df) - 1)) + 1 if "epoch" in row else len(df),
        "precision": find_metric(row, METRIC_ALIASES["precision"]),
        "recall": find_metric(row, METRIC_ALIASES["recall"]),
        "mAP50": find_metric(row, METRIC_ALIASES["map50"]),
        "mAP50_95": find_metric(row, METRIC_ALIASES["map50_95"]),
        "model_size_MB": checkpoint_size_mb(run_dir),
        "inference_ms": speed_rows.get(run_dir.name, {}).get("avg_ms_per_image"),
        "fps": speed_rows.get(run_dir.name, {}).get("fps"),
        "best_pt": str((run_dir / "weights" / "best.pt").as_posix()),
    }
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "待填"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = ["实验", "模型", "输入尺寸", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "模型大小/MB", "推理时间/ms", "FPS"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["experiment"]),
                    fmt(row["model"]),
                    fmt(row["imgsz"]),
                    fmt(row["precision"]),
                    fmt(row["recall"]),
                    fmt(row["mAP50"]),
                    fmt(row["mAP50_95"]),
                    fmt(row["model_size_MB"]),
                    fmt(row["inference_ms"]),
                    fmt(row["fps"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = [p for p in sorted(args.runs.iterdir()) if p.is_dir() and not p.name.endswith("_val")]
    speed_rows = load_speed(args.speed)
    rows = [row for run_dir in run_dirs if (row := summarize_run(run_dir, speed_rows)) is not None]
    if not rows:
        raise FileNotFoundError(f"no results.csv files found under {args.runs}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, args.markdown)
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {args.markdown.resolve()}")


if __name__ == "__main__":
    main()
