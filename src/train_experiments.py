"""Run the minimal YOLO remote-sensing experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO experiments from configs/experiments.yaml.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.yaml"))
    parser.add_argument("--data", type=Path, default=Path("configs/remote.yaml"))
    parser.add_argument("--device", default=None, help="Example: 0, cpu, mps. Leave empty for auto.")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Fast smoke test: 3 epochs and small batch.")
    parser.add_argument("--resume", action="store_true", help="Resume if possible.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_runtime_data_yaml(data: Path, project: Path) -> Path:
    """Write a data yaml with an absolute dataset path for Ultralytics."""
    cfg = load_config(data)
    dataset_root = Path(cfg["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data.parent / dataset_root).resolve()

    runtime_data = dict(cfg)
    runtime_data["path"] = dataset_root.as_posix()
    project.mkdir(parents=True, exist_ok=True)
    runtime_path = project / "_runtime_remote.yaml"
    runtime_path.write_text(
        yaml.safe_dump(runtime_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_path


def train_one(exp: dict, project: Path, data: Path, device: str | None, workers: int, quick: bool, resume: bool) -> None:
    from ultralytics import YOLO

    epochs = 3 if quick else int(exp.get("epochs", 100))
    batch = min(int(exp.get("batch", 16)), 4) if quick else int(exp.get("batch", 16))
    model_name = exp["model"]

    print(f"\n=== Training {exp['name']} ===")
    print(f"model={model_name}, imgsz={exp['imgsz']}, epochs={epochs}, batch={batch}")

    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=int(exp["imgsz"]),
        batch=batch,
        project=str(project),
        name=exp["name"],
        device=device,
        workers=workers,
        patience=20,
        pretrained=True,
        cache=False,
        resume=resume,
        plots=True,
    )

    best = project / exp["name"] / "weights" / "best.pt"
    if best.exists():
        print(f"Validating best checkpoint: {best}")
        model = YOLO(str(best))
        model.val(data=str(data), imgsz=int(exp["imgsz"]), device=device, project=str(project), name=f"{exp['name']}_val")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    project = Path(cfg.get("project", "runs/remote_yolo"))
    data = write_runtime_data_yaml(args.data, project)

    for exp in cfg["experiments"]:
        train_one(
            exp=exp,
            project=project,
            data=data,
            device=args.device,
            workers=args.workers,
            quick=args.quick,
            resume=args.resume,
        )

    print("\nTraining finished. Run:")
    print("python src/summarize_results.py --runs runs/remote_yolo --output reports/thesis_results.csv")


if __name__ == "__main__":
    main()
