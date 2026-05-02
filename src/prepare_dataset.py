"""Prepare a small remote-sensing dataset for YOLO training.

Supported inputs:
1. Existing YOLO labels:
   raw/images/*.jpg
   raw/labels/*.txt

2. Pascal VOC XML labels, used by many remote-sensing datasets such as DIOR:
   raw/images/*.jpg
   raw/annotations/*.xml

Output:
   datasets/remote_dataset/images/{train,val,test}
   datasets/remote_dataset/labels/{train,val,test}
   configs/remote.yaml
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LOGICAL_STEM_RE = re.compile(r"^(.+?_\d+)(?:_\d+)?$")


@dataclass
class SplitRatio:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1, got {total}")


def find_images(image_dir: Path) -> list[Path]:
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise FileNotFoundError(f"no images found in {image_dir}")
    return sorted(images)


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def voc_box_to_yolo(
    xmin: float, ymin: float, xmax: float, ymax: float, width: int, height: int
) -> tuple[float, float, float, float]:
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_voc_xml(
    xml_path: Path,
    image_path: Path,
    class_to_id: dict[str, int],
) -> str:
    root = ET.parse(xml_path).getroot()
    width, height = image_size(image_path)
    lines: list[str] = []

    for obj in root.findall("object"):
        name_node = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_node is None or bndbox is None:
            continue

        class_name = (name_node.text or "").strip()
        if class_name not in class_to_id:
            continue

        values = {}
        for key in ("xmin", "ymin", "xmax", "ymax"):
            node = bndbox.find(key)
            if node is None or node.text is None:
                break
            values[key] = float(node.text)
        else:
            x, y, w, h = voc_box_to_yolo(
                values["xmin"], values["ymin"], values["xmax"], values["ymax"], width, height
            )
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
            w = min(max(w, 0.0), 1.0)
            h = min(max(h, 0.0), 1.0)
            if w > 0 and h > 0:
                lines.append(f"{class_to_id[class_name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines) + ("\n" if lines else "")


def logical_stem(stem: str) -> str:
    """Map names like oiltank_102_20230301212336 to oiltank_102."""
    match = LOGICAL_STEM_RE.match(stem)
    return match.group(1) if match else stem


def find_annotation(annotation_dir: Path, image_stem: str) -> Path | None:
    exact = annotation_dir / f"{image_stem}.xml"
    if exact.exists():
        return exact

    key = logical_stem(image_stem)
    candidates = sorted(annotation_dir.glob(f"{key}*.xml"))
    return candidates[0] if candidates else None


def split_items(items: list[Path], ratio: SplitRatio, seed: int) -> dict[str, list[Path]]:
    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * ratio.train)
    val_end = train_end + int(n * ratio.val)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_dataset_yaml(output_root: Path, names: list[str], yaml_path: Path) -> None:
    rel_path = Path("../datasets") / output_root.name
    data = {
        "path": rel_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names)},
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def copy_or_convert(
    image: Path,
    split: str,
    image_dir: Path,
    label_dir: Path | None,
    annotation_dir: Path | None,
    output_root: Path,
    class_to_id: dict[str, int],
) -> bool:
    out_image_dir = output_root / "images" / split
    out_label_dir = output_root / "labels" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    rel = image.relative_to(image_dir)
    output_image = out_image_dir / rel.name
    output_label = out_label_dir / f"{image.stem}.txt"
    shutil.copy2(image, output_image)

    if label_dir is not None:
        source_label = label_dir / f"{image.stem}.txt"
        if source_label.exists():
            shutil.copy2(source_label, output_label)
            return True
        output_label.write_text("", encoding="utf-8")
        return False

    if annotation_dir is not None:
        xml_path = find_annotation(annotation_dir, image.stem)
        if xml_path is not None:
            output_label.write_text(convert_voc_xml(xml_path, image, class_to_id), encoding="utf-8")
            return True
        output_label.write_text("", encoding="utf-8")
        return False

    output_label.write_text("", encoding="utf-8")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits.")
    parser.add_argument("--images", required=True, type=Path, help="Input image directory.")
    parser.add_argument("--labels", type=Path, help="Existing YOLO label directory.")
    parser.add_argument("--annotations", type=Path, help="Pascal VOC XML annotation directory.")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["aircraft", "oiltank", "overpass", "playground"],
        help="Class names to keep, in YOLO class-id order.",
    )
    parser.add_argument("--output", type=Path, default=Path("datasets/remote_dataset"))
    parser.add_argument("--yaml", type=Path, default=Path("configs/remote.yaml"))
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true", help="Delete output dataset before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.labels and args.annotations:
        raise ValueError("use either --labels or --annotations, not both")
    if not args.labels and not args.annotations:
        raise ValueError("provide --labels for YOLO labels or --annotations for VOC XML labels")

    ratio = SplitRatio(args.train, args.val, args.test)
    ratio.validate()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    class_to_id = {name: i for i, name in enumerate(args.classes)}
    images = find_images(args.images)
    splits = split_items(images, ratio, args.seed)

    labeled_count = 0
    total_count = 0
    for split, split_images in splits.items():
        for image in tqdm(split_images, desc=f"prepare {split}"):
            has_label = copy_or_convert(
                image=image,
                split=split,
                image_dir=args.images,
                label_dir=args.labels,
                annotation_dir=args.annotations,
                output_root=args.output,
                class_to_id=class_to_id,
            )
            labeled_count += int(has_label)
            total_count += 1

    write_dataset_yaml(args.output, args.classes, args.yaml)
    print(f"Prepared {total_count} images, {labeled_count} with labels.")
    print(f"Dataset: {args.output.resolve()}")
    print(f"YAML: {args.yaml.resolve()}")


if __name__ == "__main__":
    main()
