"""Export trained YOLO checkpoints for deployment experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export best.pt checkpoints to ONNX.")
    parser.add_argument("--runs", type=Path, default=Path("runs/remote_yolo"))
    parser.add_argument("--data", type=Path, default=Path("configs/remote.yaml"))
    parser.add_argument("--format", default="onnx", choices=["onnx", "openvino", "engine", "torchscript"])
    parser.add_argument("--int8", action="store_true", help="INT8 export, requires calibration data support.")
    parser.add_argument("--half", action="store_true", help="FP16 export where supported.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def write_runtime_data_yaml(data: Path, output_dir: Path) -> Path:
    with data.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_root = Path(cfg["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data.parent / dataset_root).resolve()

    runtime_data = dict(cfg)
    runtime_data["path"] = dataset_root.as_posix()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / "_runtime_remote.yaml"
    runtime_path.write_text(
        yaml.safe_dump(runtime_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_path


def main() -> None:
    args = parse_args()
    checkpoints = sorted(args.runs.glob("*/weights/best.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found under {args.runs}")

    from ultralytics import YOLO

    data = write_runtime_data_yaml(args.data, args.runs)
    for checkpoint in checkpoints:
        print(f"\n=== Export {checkpoint} ===")
        model = YOLO(str(checkpoint))
        model.export(
            format=args.format,
            data=str(data),
            int8=args.int8,
            half=args.half,
            device=args.device,
            simplify=True,
        )

    print("\nExport finished.")


if __name__ == "__main__":
    main()
