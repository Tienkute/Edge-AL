"""
Kiểm tra tính hợp lệ và toàn vẹn của dataset cho cuộc thi EDGE AI CHALLENGE 2026 – Warm-up.
"""

import csv
import sys
from pathlib import Path

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def check_dataset():
    # Xác định đường dẫn gốc của project
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"

    errors = []

    # 1. Kiểm tra thư mục tồn tại
    train_images_dir = raw_dir / "train" / "images"
    train_labels_dir = raw_dir / "train" / "labels"
    test_images_dir = raw_dir / "test" / "images"
    classes_file = raw_dir / "classes.txt"
    intent_train_file = raw_dir / "intent_train.csv"
    intent_test_file = raw_dir / "intent_test.csv"
    sample_sub_file = raw_dir / "sample_submission.csv"

    for path, desc in [
        (train_images_dir, "Train images directory"),
        (train_labels_dir, "Train labels directory"),
        (test_images_dir, "Test images directory"),
        (classes_file, "Classes file"),
        (intent_train_file, "Intent train CSV"),
        (intent_test_file, "Intent test CSV"),
        (sample_sub_file, "Sample submission CSV"),
    ]:
        if not path.exists():
            errors.append(f"Không tìm thấy: {desc} ({path})")

    if errors:
        print("=== DATASET CHECK FAILED ===")
        for err in errors:
            print(f"[-] {err}")
        return False

    # 2. Kiểm tra ảnh và nhãn train
    image_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    train_images = [f for f in train_images_dir.iterdir() if f.suffix.lower() in image_exts]
    train_labels = [f for f in train_labels_dir.iterdir() if f.suffix.lower() == ".txt"]

    train_img_stems = {f.stem: f for f in train_images}
    train_lbl_stems = {f.stem: f for f in train_labels}

    missing_labels = []
    for stem in train_img_stems:
        if stem not in train_lbl_stems:
            missing_labels.append(stem)

    missing_images = []
    for stem in train_lbl_stems:
        if stem not in train_img_stems:
            missing_images.append(stem)

    # 3. Kiểm tra ảnh test
    test_images = [f for f in test_images_dir.iterdir() if f.suffix.lower() in image_exts]

    # 4. Kiểm tra classes.txt
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    # 5. Kiểm tra intent_train.csv
    intent_train_rows = []
    intent_train_dup_ids = set()
    with open(intent_train_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        expected_intent_train_cols = ["question_id", "question", "intent"]
        if header != expected_intent_train_cols:
            errors.append(
                f"Header intent_train.csv không đúng: {header} (Kỳ vọng: {expected_intent_train_cols})"
            )
        seen_ids = set()
        for row in reader:
            if not row:
                continue
            qid = row[0]
            if qid in seen_ids:
                intent_train_dup_ids.add(qid)
            seen_ids.add(qid)
            intent_train_rows.append(row)

    if intent_train_dup_ids:
        errors.append(f"intent_train.csv có question_id bị trùng: {intent_train_dup_ids}")

    # 6. Kiểm tra intent_test.csv
    intent_test_rows = []
    intent_test_dup_ids = set()
    with open(intent_test_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        expected_intent_test_cols = ["question_id", "question"]
        if header != expected_intent_test_cols:
            errors.append(
                f"Header intent_test.csv không đúng: {header} (Kỳ vọng: {expected_intent_test_cols})"
            )
        seen_ids = set()
        for row in reader:
            if not row:
                continue
            qid = row[0]
            if qid in seen_ids:
                intent_test_dup_ids.add(qid)
            seen_ids.add(qid)
            intent_test_rows.append(row)

    if intent_test_dup_ids:
        errors.append(f"intent_test.csv có question_id bị trùng: {intent_test_dup_ids}")

    # 7. Kiểm tra sample_submission.csv
    sample_sub_rows = []
    with open(sample_sub_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        expected_sub_cols = ["task_type", "sample_id", "prediction"]
        if header != expected_sub_cols:
            errors.append(
                f"Header sample_submission.csv không đúng: {header} (Kỳ vọng: {expected_sub_cols})"
            )
        seen_sample_pairs = set()
        for row in reader:
            if not row:
                continue
            if len(row) >= 2:
                pair = (row[0], row[1])
                if pair in seen_sample_pairs:
                    errors.append(f"sample_submission.csv có cặp trùng lặp: {pair}")
                seen_sample_pairs.add(pair)
            sample_sub_rows.append(row)

    # 8. Kiểm tra định dạng từng dòng YOLO label
    invalid_labels = []
    tolerance = 1e-5
    for lbl_file in train_labels:
        with open(lbl_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        file_invalid = False
        for line_idx, line in enumerate(lines, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) != 5:
                invalid_labels.append(
                    f"{lbl_file.name}:{line_idx} - Không đủ 5 phần tử: '{line_str}'"
                )
                file_invalid = True
                break
            
            # Kiểm tra class_id
            try:
                cid = int(parts[0])
                if not (0 <= cid <= 9):
                    invalid_labels.append(
                        f"{lbl_file.name}:{line_idx} - class_id {cid} nằm ngoài [0, 9]"
                    )
                    file_invalid = True
                    break
            except ValueError:
                invalid_labels.append(
                    f"{lbl_file.name}:{line_idx} - class_id không phải số nguyên: '{parts[0]}'"
                )
                file_invalid = True
                break

            # Kiểm tra cx, cy, w, h
            coords_valid = True
            for coord_idx, coord_name in enumerate(["cx", "cy", "w", "h"], start=1):
                try:
                    val = float(parts[coord_idx])
                    if not (0.0 - tolerance <= val <= 1.0 + tolerance):
                        invalid_labels.append(
                            f"{lbl_file.name}:{line_idx} - {coord_name}={val} nằm ngoài [0, 1]"
                        )
                        coords_valid = False
                        break
                except ValueError:
                    invalid_labels.append(
                        f"{lbl_file.name}:{line_idx} - {coord_name} không phải float: '{parts[coord_idx]}'"
                    )
                    coords_valid = False
                    break
            
            if not coords_valid:
                file_invalid = True
                break

    # Kiểm tra số lượng theo mô tả
    if len(train_images) != 140:
        errors.append(f"Train images: {len(train_images)} (Kỳ vọng: 140)")
    if len(train_labels) != 140:
        errors.append(f"Train labels: {len(train_labels)} (Kỳ vọng: 140)")
    if missing_labels:
        errors.append(f"Ảnh train thiếu file label tương ứng ({len(missing_labels)}): {missing_labels}")
    if missing_images:
        errors.append(f"Label train không có file ảnh tương ứng ({len(missing_images)}): {missing_images}")
    if len(test_images) != 60:
        errors.append(f"Test images: {len(test_images)} (Kỳ vọng: 60)")
    if len(classes) != 10:
        errors.append(f"Classes: {len(classes)} (Kỳ vọng: 10)")
    if len(intent_train_rows) != 63:
        errors.append(f"Intent train rows: {len(intent_train_rows)} (Kỳ vọng: 63)")
    if len(intent_test_rows) != 28:
        errors.append(f"Intent test rows: {len(intent_test_rows)} (Kỳ vọng: 28)")
    if invalid_labels:
        errors.append(f"Có {len(invalid_labels)} file/dòng label không hợp lệ")

    is_ok = len(errors) == 0 and len(missing_labels) == 0 and len(invalid_labels) == 0

    print("=== DATASET CHECK ===")
    print(f"Train images: {len(train_images)}")
    print(f"Train labels: {len(train_labels)}")
    print(f"Test images: {len(test_images)}")
    print(f"Classes: {len(classes)}")
    print(f"Intent train: {len(intent_train_rows)}")
    print(f"Intent test: {len(intent_test_rows)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Invalid labels: {len(invalid_labels)}")
    print(f"Dataset status: {'OK' if is_ok else 'FAILED'}")

    if not is_ok:
        print("\nChi tiết lỗi / cảnh báo:")
        for err in errors:
            print(f"[-] {err}")
        if invalid_labels:
            print("\nChi tiết label không hợp lệ (tối đa 10 dòng đầu):")
            for inv in invalid_labels[:10]:
                print(f"    {inv}")

    return is_ok


if __name__ == "__main__":
    success = check_dataset()
    sys.exit(0 if success else 1)
