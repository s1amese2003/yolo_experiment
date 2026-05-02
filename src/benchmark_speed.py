"""Benchmark trained YOLO checkpoints on validation/test images."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure average inference latency for best.pt checkpoints.")
    parser.add_argument("--runs", type=Path, default=Path("runs/remote_yolo"))
    parser.add_argument("--data", type=Path, default=Path("configs/remote.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/speed_results.csv"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    cfg = load_yaml(data_yaml)
    root = (data_yaml.parent / cfg["path"]).resolve()
    split_dir = root / cfg[split]
    images = sorted(p for p in split_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise FileNotFoundError(f"no images found in {split_dir}")
    return images


def cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_model(checkpoint: Path, images: list[Path], samples: int, warmup: int, device: str | None) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    picked = random.Random(42).sample(images, k=min(samples, len(images)))
    warmup_images = picked[: min(warmup, len(picked))]

    for image in warmup_images:
        model.predict(source=str(image), device=device, verbose=False)

    cuda_sync()
    start = time.perf_counter()
    for image in picked:
        model.predict(source=str(image), device=device, verbose=False)
    cuda_sync()
    elapsed = time.perf_counter() - start

    return {
        "experiment": checkpoint.parents[1].name,
        "checkpoint": str(checkpoint.as_posix()),
        "samples": len(picked),
        "total_seconds": elapsed,
        "avg_ms_per_image": elapsed / len(picked) * 1000,
        "fps": len(picked) / elapsed if elapsed > 0 else None,
    }


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.runs.glob("*/weights/best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {args.runs}")

    images = resolve_split_images(args.data, args.split)
    rows = []
    for checkpoint in checkpoints:
        print(f"Benchmarking {checkpoint}")
        rows.append(benchmark_model(checkpoint, images, args.samples, args.warmup, args.device))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
