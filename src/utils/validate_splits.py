"""
Kiểm tra tính hợp lệ của các split Detection và Intent.
EDGE AI CHALLENGE 2026 – Warm-up

Kiểm tra Detection:
  - Tổng 140 ảnh, không trùng, không thiếu
  - Không có ảnh test trong manifest
  - Tất cả file label tồn tại
  - 10 class đều có trong train
  - Phân bố class

Kiểm tra Intent:
  - Tổng 63 câu, đúng 3 fold
  - Mỗi question_id xuất hiện đúng một lần
  - Không có question_id từ intent_test.csv
  - Không trùng ID
  - Mỗi fold có đủ nhãn cần thiết
  - Cảnh báo nếu OUT_OF_SCOPE vắng mặt trong val

Nếu phát hiện ảnh gần trùng giữa train và val → cảnh báo rõ ràng.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Đảm bảo Unicode trên Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_utils import load_classes, load_yolo_label


def validate_detection():
    """
    Kiểm tra toàn bộ Detection split.
    Trả về (is_valid: bool, warnings: list[str]).
    """
    data_raw = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    train_images_dir = data_raw / "train" / "images"
    train_labels_dir = data_raw / "train" / "labels"
    test_images_dir = data_raw / "test" / "images"
    classes_file = data_raw / "classes.txt"

    det_train_path = processed_dir / "detection_train.txt"
    det_val_path = processed_dir / "detection_val.txt"

    errors = []
    warnings = []

    print("\n[DETECTION] Kiểm tra split...")

    # Kiểm tra file manifest tồn tại
    for p in [det_train_path, det_val_path]:
        if not p.exists():
            errors.append(f"File manifest không tồn tại: {p}")
    if errors:
        for e in errors: print(f"  [-] LỖI: {e}")
        return False, warnings

    # Đọc manifest
    with open(det_train_path, "r", encoding="utf-8") as f:
        train_names = [line.strip() for line in f if line.strip()]
    with open(det_val_path, "r", encoding="utf-8") as f:
        val_names = [line.strip() for line in f if line.strip()]

    n_train = len(train_names)
    n_val = len(val_names)
    n_total = n_train + n_val

    print(f"  Train: {n_train} ảnh | Val: {n_val} ảnh | Tổng: {n_total}")

    # Kiểm tra tổng số ảnh
    if n_total != 140:
        errors.append(f"Tổng số ảnh phải là 140, nhưng có: {n_total}")

    # Kiểm tra không trùng
    train_set = set(train_names)
    val_set = set(val_names)
    overlap = train_set & val_set
    if overlap:
        errors.append(f"Có {len(overlap)} ảnh xuất hiện ở cả hai tập: {sorted(overlap)}")
    else:
        print("  [+] Không có ảnh trùng lặp giữa train và val.")

    # Kiểm tra trùng tên trong cùng một file
    if len(train_names) != len(train_set):
        errors.append("detection_train.txt có tên ảnh bị trùng!")
    if len(val_names) != len(val_set):
        errors.append("detection_val.txt có tên ảnh bị trùng!")

    # Kiểm tra không có ảnh test
    test_image_names = {f.name for f in test_images_dir.iterdir() if f.suffix.lower() == ".png"}
    test_in_train = train_set & test_image_names
    test_in_val = val_set & test_image_names
    if test_in_train:
        errors.append(f"Ảnh TEST xuất hiện trong train manifest: {sorted(test_in_train)}")
    if test_in_val:
        errors.append(f"Ảnh TEST xuất hiện trong val manifest: {sorted(test_in_val)}")
    if not test_in_train and not test_in_val:
        print("  [+] Không có ảnh test nào trong manifest.")

    # Kiểm tra file ảnh và label tồn tại
    missing_imgs = []
    missing_labels = []
    for name in train_names + val_names:
        img_path = train_images_dir / name
        lbl_path = train_labels_dir / f"{Path(name).stem}.txt"
        if not img_path.exists():
            missing_imgs.append(name)
        if not lbl_path.exists():
            missing_labels.append(name)
    if missing_imgs:
        errors.append(f"{len(missing_imgs)} ảnh không tồn tại: {missing_imgs}")
    else:
        print("  [+] Tất cả file ảnh tồn tại.")
    if missing_labels:
        errors.append(f"{len(missing_labels)} file label không tồn tại: {missing_labels}")
    else:
        print("  [+] Tất cả file label tồn tại.")

    # Kiểm tra phân bố class
    classes = load_classes(classes_file)
    n_classes = len(classes)

    def get_class_coverage(names_list):
        coverage = set()
        for name in names_list:
            lbl_path = train_labels_dir / f"{Path(name).stem}.txt"
            for cid, *_ in load_yolo_label(lbl_path):
                if 0 <= cid < n_classes:
                    coverage.add(cid)
        return coverage

    train_coverage = get_class_coverage(train_names)
    val_coverage = get_class_coverage(val_names)

    missing_in_train = [classes[c] for c in range(n_classes) if c not in train_coverage]
    missing_in_val = [classes[c] for c in range(n_classes) if c not in val_coverage]

    if missing_in_train:
        errors.append(f"Class không có trong TRAIN: {missing_in_train}")
    else:
        print(f"  [+] Tất cả {n_classes} class có trong train.")

    if missing_in_val:
        warnings.append(f"Class không có trong VAL: {missing_in_val}")
        print(f"  [!] CẢNH BÁO: Class không có trong val: {missing_in_val}")
    else:
        print(f"  [+] Tất cả {n_classes} class có trong val.")

    # Kiểm tra ảnh gần trùng (pHash) giữa train và val
    print("  Kiểm tra near-duplicate giữa train và val (pHash)...")
    try:
        from scipy.fft import dct as scipy_dct
        from PIL import Image

        def compute_phash_validate(img_path, hash_size=8, highfreq=4):
            img_size = hash_size * highfreq
            with Image.open(img_path) as img:
                small = np.array(img.convert("L").resize((img_size, img_size), Image.LANCZOS), dtype=float)
            dct_coef = scipy_dct(scipy_dct(small, axis=0), axis=1)
            top_left = dct_coef[:hash_size, :hash_size]
            return (top_left > np.median(top_left)).flatten()

        # Chỉ tính pHash cho train và val images
        train_hashes = {}
        for name in train_names:
            try:
                train_hashes[name] = compute_phash_validate(train_images_dir / name)
            except Exception:
                pass

        val_hashes = {}
        for name in val_names:
            try:
                val_hashes[name] = compute_phash_validate(train_images_dir / name)
            except Exception:
                pass

        # Kiểm tra cross-set pairs
        near_dup_cross = []
        for t_name, t_hash in train_hashes.items():
            for v_name, v_hash in val_hashes.items():
                dist = int(np.sum(t_hash != v_hash))
                if dist <= 10:
                    near_dup_cross.append((t_name, v_name, dist))

        if near_dup_cross:
            warnings.append(f"CẢNH BÁO NEAR-DUPLICATE CROSS-SET: {near_dup_cross}")
            print(f"  [!] CẢNH BÁO: {len(near_dup_cross)} cặp ảnh gần trùng giữa train và val:")
            for t, v, d in near_dup_cross:
                print(f"       TRAIN:{t} <-> VAL:{v} (Hamming={d})")
        else:
            print("  [+] Không phát hiện ảnh gần trùng giữa train và val (pHash Hamming <= 10).")

    except ImportError:
        warnings.append("scipy không có để kiểm tra pHash. Bỏ qua bước near-dup cross-set.")
        print("  [!] scipy không có, bỏ qua kiểm tra pHash cross-set.")

    is_valid = len(errors) == 0
    return is_valid, warnings, errors, n_train, n_val


def validate_intent():
    """
    Kiểm tra toàn bộ Intent split.
    Trả về (is_valid: bool, warnings: list[str], errors: list[str]).
    """
    data_raw = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    intent_train_path = data_raw / "intent_train.csv"
    intent_test_path = data_raw / "intent_test.csv"
    folds_csv_path = processed_dir / "intent_folds.csv"

    errors = []
    warnings = []

    print("\n[INTENT] Kiểm tra split...")

    # Kiểm tra file tồn tại
    if not folds_csv_path.exists():
        errors.append(f"File intent_folds.csv không tồn tại: {folds_csv_path}")
        for e in errors: print(f"  [-] LỖI: {e}")
        return False, warnings, errors

    # Đọc dữ liệu
    df_train_original = pd.read_csv(intent_train_path)
    df_test = pd.read_csv(intent_test_path)
    df_folds = pd.read_csv(folds_csv_path)

    n_train_orig = len(df_train_original)
    n_folds_rows = len(df_folds)

    print(f"  intent_train.csv: {n_train_orig} câu")
    print(f"  intent_folds.csv: {n_folds_rows} hàng")

    # Kiểm tra tổng số câu
    if n_folds_rows != 63:
        errors.append(f"intent_folds.csv phải có 63 hàng, nhưng có: {n_folds_rows}")

    # Kiểm tra đúng 3 fold
    unique_folds = sorted(df_folds["fold"].unique())
    if list(unique_folds) != [0, 1, 2]:
        errors.append(f"Fold values phải là [0, 1, 2], nhưng có: {unique_folds}")
    else:
        print("  [+] Có đúng 3 fold (0, 1, 2).")

    # Kiểm tra mỗi question_id xuất hiện đúng 1 lần
    dup_ids = df_folds[df_folds["question_id"].duplicated()]["question_id"].tolist()
    if dup_ids:
        errors.append(f"Có {len(dup_ids)} question_id trùng lặp trong folds: {dup_ids}")
    else:
        print("  [+] Không có question_id trùng lặp.")

    # Kiểm tra không có question_id từ intent_test
    test_ids = set(df_test["question_id"].values)
    fold_ids = set(df_folds["question_id"].values)
    test_in_folds = fold_ids & test_ids
    if test_in_folds:
        errors.append(f"question_id từ intent_test xuất hiện trong folds: {test_in_folds}")
    else:
        print("  [+] Không có question_id test trong folds.")

    # Merge để lấy intent
    df_merged = df_folds.merge(df_train_original[["question_id", "intent"]], on="question_id", how="left")
    missing_intent = df_merged["intent"].isna().sum()
    if missing_intent > 0:
        errors.append(f"{missing_intent} question_id trong folds không có trong intent_train.csv")

    # Kiểm tra từng fold
    intents_all = sorted(df_train_original["intent"].unique())
    print(f"\n  Phân bố từng fold:")
    for fold_idx in range(3):
        val_df = df_merged[df_merged["fold"] == fold_idx]
        train_df = df_merged[df_merged["fold"] != fold_idx]
        n_val = len(val_df)
        n_train_fold = len(train_df)
        print(f"\n  Fold {fold_idx}: Train={n_train_fold}, Val={n_val}")

        val_intents = set(val_df["intent"].dropna().values)
        train_intents = set(train_df["intent"].dropna().values)

        # Kiểm tra OUT_OF_SCOPE
        if "OUT_OF_SCOPE" not in val_intents:
            msg = f"Fold {fold_idx} VAL không có OUT_OF_SCOPE!"
            warnings.append(msg)
            print(f"    [!] CẢNH BÁO: {msg}")
        else:
            print(f"    [+] Fold {fold_idx} VAL có OUT_OF_SCOPE.")

        # In phân bố
        print(f"  {'Intent':<18} {'Train':<8} {'Val':<6}")
        print("  " + "-" * 34)
        for intent in intents_all:
            tc = int((train_df["intent"] == intent).sum())
            vc = int((val_df["intent"] == intent).sum())
            print(f"  {intent:<18} {tc:<8} {vc:<6}")

    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def main():
    print("=" * 60)
    print("PHASE 3A: KIỂM TRA TÍNH HỢP LỆ CÁC SPLIT")
    print("=" * 60)

    all_errors = []
    all_warnings = []

    # --- Detection ---
    det_result = validate_detection()
    det_valid, det_warnings, det_errors, n_train, n_val = det_result
    all_errors.extend(det_errors)
    all_warnings.extend(det_warnings)

    # --- Intent ---
    int_result = validate_intent()
    int_valid, int_warnings, int_errors = int_result
    all_errors.extend(int_errors)
    all_warnings.extend(int_warnings)

    # --- Tóm tắt ---
    print("\n" + "=" * 60)
    print("KẾT QUẢ KIỂM TRA TỔNG HỢP")
    print("=" * 60)
    print(f"Tổng lỗi:       {len(all_errors)}")
    print(f"Tổng cảnh báo:  {len(all_warnings)}")

    if all_errors:
        print("\nCác lỗi cần sửa:")
        for e in all_errors:
            print(f"  [-] {e}")
    else:
        print("\n[OK] Tất cả kiểm tra PASSED - không có lỗi.")

    if all_warnings:
        print("\nCảnh báo (không ngăn chạy, nhưng cần xem xét):")
        for w in all_warnings:
            print(f"  [!] {w}")

    # Exit code
    sys.exit(0 if not all_errors else 1)


if __name__ == "__main__":
    main()

