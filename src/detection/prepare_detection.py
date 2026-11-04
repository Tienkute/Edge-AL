"""EDGE AI Challenge 2026 – Phase 4A: build YOLO dataset manifests.

This script never modifies data/raw, train/val split manifests, or prior models.
Run:
  python src/detection/prepare_detection.py --check-only
  python src/detection/prepare_detection.py --prepare

The generated train/val image lists point directly to the existing raw images.
Ultralytics resolves corresponding labels by replacing /images/ with /labels/.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import math
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
YOLO_DIR = PROCESSED / "yolo"
TRAIN_IMAGES = RAW / "train" / "images"
TRAIN_LABELS = RAW / "train" / "labels"
EXPECTED_CLASSES = 10
EXPECTED_TRAIN = 112
EXPECTED_VAL = 28


def read_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay manifest: {path}")
    names: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Manifests from Phase 3A should list image names, e.g. 0001.png.
        name = Path(line.replace("\\", "/")).name
        if not name.lower().endswith(".png"):
            raise ValueError(f"{path.name}:{line_number}: khong phai PNG: {line!r}")
        if name in names:
            raise ValueError(f"{path.name}:{line_number}: anh bi lap: {name}")
        names.append(name)
    return names


def read_classes() -> list[str]:
    path = RAW / "classes.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    classes = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(classes) != EXPECTED_CLASSES or len(set(classes)) != EXPECTED_CLASSES:
        raise ValueError(f"classes.txt can {EXPECTED_CLASSES} nhan rieng biet; nhan duoc {len(classes)}")
    return classes


def check_label(path: Path, num_classes: int) -> Counter[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Thieu nhan YOLO: {path}")
    counts: Counter[int] = Counter()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path.name}:{number}: can 5 cot, co {len(parts)}")
        try:
            cls_float = float(parts[0])
            cls = int(cls_float)
            cx, cy, w, h = map(float, parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path.name}:{number}: khong phai so hop le") from exc
        if cls_float != cls or not 0 <= cls < num_classes:
            raise ValueError(f"{path.name}:{number}: class_id sai: {parts[0]}")
        if not all(math.isfinite(v) for v in (cx, cy, w, h)):
            raise ValueError(f"{path.name}:{number}: co NaN/Inf")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path.name}:{number}: bbox normalized ngoai 0..1")
        # Check if bbox corners stay within image, allowing rounding noise.
        eps = 1e-4
        if cx - w / 2 < -eps or cy - h / 2 < -eps or cx + w / 2 > 1 + eps or cy + h / 2 > 1 + eps:
            raise ValueError(f"{path.name}:{number}: bbox vuot ngoai bien anh")
        counts[cls] += 1
    return counts


def analyze() -> tuple[list[str], list[str], list[str]]:
    classes = read_classes()
    train = read_manifest(PROCESSED / "detection_train.txt")
    val = read_manifest(PROCESSED / "detection_val.txt")
    if len(train) != EXPECTED_TRAIN or len(val) != EXPECTED_VAL:
        raise ValueError(f"Sai split: train={len(train)} (can {EXPECTED_TRAIN}), val={len(val)} (can {EXPECTED_VAL})")
    if set(train) & set(val):
        raise ValueError(f"Anh bi trung 2 tap: {sorted(set(train) & set(val))}")
    available = {p.name for p in TRAIN_IMAGES.glob("*.png")}
    listed = set(train) | set(val)
    if available != listed:
        raise ValueError(f"Anh thieu trong manifests: {sorted(available-listed)}; ten khong co file: {sorted(listed-available)}")
    if len(available) != EXPECTED_TRAIN + EXPECTED_VAL:
        raise ValueError(f"So anh train goc bat thuong: {len(available)}")
    counts_train: Counter[int] = Counter()
    counts_val: Counter[int] = Counter()
    for split, names, result_counts in (("train", train, counts_train), ("val", val, counts_val)):
        for name in names:
            image_path = TRAIN_IMAGES / name
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            result_counts.update(check_label(TRAIN_LABELS / f"{Path(name).stem}.txt", len(classes)))
    print("=== EDGE AI 2026: PHASE 4A / KIEM TRA DATASET YOLO ===")
    print(f"Project: {ROOT}")
    print(f"Train: {len(train)} anh | Val: {len(val)} anh | Classes: {len(classes)}")
    print("ID | Class                     | Train boxes | Val boxes")
    for i, name in enumerate(classes):
        print(f"{i:2} | {name:25} | {counts_train[i]:11} | {counts_val[i]:9}")
    print(f"Tong boxes: {sum(counts_train.values()) + sum(counts_val.values())}")
    if sum(counts_train.values()) + sum(counts_val.values()) != 728:
        print("CANH BAO: Khac 728 boxes so voi bao cao EDA; hay xem lai split/labels.")
    print("CHECK OK: khong sua file raw, khong train model.")
    return train, val, classes


def prepare(train: list[str], val: list[str], classes: list[str]) -> None:
    # Never silently overwrite previously generated data.
    targets = [YOLO_DIR / "train_images.txt", YOLO_DIR / "val_images.txt", YOLO_DIR / "dataset.yaml"]
    existing = [p for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"File output da ton tai; khong ghi de: {existing}")
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "train_images.txt").write_text(
        "\n".join((TRAIN_IMAGES / name).as_posix() for name in train) + "\n", encoding="utf-8"
    )
    (YOLO_DIR / "val_images.txt").write_text(
        "\n".join((TRAIN_IMAGES / name).as_posix() for name in val) + "\n", encoding="utf-8"
    )
    yaml_data = {
        "path": ROOT.as_posix(),
        "train": "data/processed/yolo/train_images.txt",
        "val": "data/processed/yolo/val_images.txt",
        "names": {i: name for i, name in enumerate(classes)},
    }
    (YOLO_DIR / "dataset.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print("\nDA TAO 3 FILE (khong copy anh, khong sua nhan):")
    for path in targets:
        print(f"  {path.relative_to(ROOT)}")
    print("PHASE 4A PREPARE OK: ban co the dung dataset.yaml cho YOLO baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate/build YOLO paths for Phase 4A")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    train, val, classes = analyze()
    if args.prepare:
        prepare(train, val, classes)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"LOI: {exc}", file=sys.stderr)
        sys.exit(1)
