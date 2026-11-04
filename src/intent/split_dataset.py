"""
Chia tập dữ liệu Intent Classification thành 3 fold Train/Validation.
EDGE AI CHALLENGE 2026 – Warm-up

Phương pháp:
  - StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
  - Giữ nguyên toàn bộ 63 câu hỏi gốc, không thay đổi nhãn.
  - Mỗi câu được dùng làm validation đúng một lần qua 3 fold.
  - KHÔNG đọc intent_test.csv để lựa chọn fold hoặc điều chỉnh nhãn.

Lưu:
  data/processed/intent_folds.csv      (question_id,fold)
  outputs/eda/intent_folds_report.csv  (thống kê từng fold)
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Đảm bảo Unicode trên Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    data_raw = PROJECT_ROOT / "data" / "raw"
    intent_train_path = data_raw / "intent_train.csv"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    eda_dir = PROJECT_ROOT / "outputs" / "eda"
    processed_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3A: CHIA DỮ LIỆU INTENT CLASSIFICATION (3-FOLD CV)")
    print("=" * 60)

    # 1. Đọc dữ liệu train
    df = pd.read_csv(intent_train_path)
    n = len(df)
    print(f"\n[1] Tổng số câu hỏi train: {n}")
    print(f"    Phân bố intent ban đầu:")
    intent_counts = df["intent"].value_counts().sort_index()
    for intent, count in intent_counts.items():
        print(f"      {intent}: {count} ({count/n*100:.1f}%)")

    intents = sorted(df["intent"].unique())
    print(f"\n    Danh sách intent ({len(intents)} loại): {intents}")

    # 2. Áp dụng StratifiedKFold
    # Encode intent thành số để StratifiedKFold hoạt động
    intent_to_id = {intent: i for i, intent in enumerate(intents)}
    y = df["intent"].map(intent_to_id).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    df["fold"] = -1  # Khởi tạo

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, y)):
        df.loc[val_idx, "fold"] = fold_idx

    # Kiểm tra: mỗi câu được gán đúng một fold
    assert (df["fold"] >= 0).all(), "Có câu chưa được gán fold!"
    assert df["fold"].isin([0, 1, 2]).all(), "Fold không hợp lệ!"
    assert len(df["fold"].unique()) == 3, "Phải có đúng 3 fold!"
    print("\n[2] Đã chia 3 fold thành công.")

    # 3. Thống kê chi tiết từng fold
    print("\n[3] Thống kê từng fold:")
    report_rows = []

    for fold_idx in range(3):
        val_mask = df["fold"] == fold_idx
        train_mask = ~val_mask
        df_val = df[val_mask]
        df_train_fold = df[train_mask]

        n_val = len(df_val)
        n_train_fold = len(df_train_fold)

        print(f"\n  Fold {fold_idx}: Train={n_train_fold}, Val={n_val}")

        val_intent_counts = df_val["intent"].value_counts().sort_index()
        train_intent_counts = df_train_fold["intent"].value_counts().sort_index()

        # Kiểm tra OUT_OF_SCOPE
        has_oos_val = "OUT_OF_SCOPE" in df_val["intent"].values
        has_oos_train = "OUT_OF_SCOPE" in df_train_fold["intent"].values
        if not has_oos_val:
            print(f"    [!] CẢNH BÁO: Fold {fold_idx} VAL không có OUT_OF_SCOPE!")
        if not has_oos_train:
            print(f"    [!] CẢNH BÁO: Fold {fold_idx} TRAIN không có OUT_OF_SCOPE!")

        print(f"  {'Intent':<18} {'Train':<8} {'Val':<6} {'Val%':<8}")
        print("  " + "-" * 42)
        for intent in intents:
            tc = int(train_intent_counts.get(intent, 0))
            vc = int(val_intent_counts.get(intent, 0))
            total_c = tc + vc
            pct = f"{vc/total_c*100:.1f}%" if total_c > 0 else "N/A"
            print(f"  {intent:<18} {tc:<8} {vc:<6} {pct:<8}")

            report_rows.append({
                "fold": fold_idx,
                "split": "val",
                "intent": intent,
                "count": vc,
            })
            report_rows.append({
                "fold": fold_idx,
                "split": "train",
                "intent": intent,
                "count": tc,
            })

    # 4. Lưu intent_folds.csv
    folds_csv_path = processed_dir / "intent_folds.csv"
    df[["question_id", "fold"]].to_csv(folds_csv_path, index=False, encoding="utf-8")
    print(f"\n[+] Đã lưu: {folds_csv_path}")

    # 5. Lưu report CSV
    report_csv_path = eda_dir / "intent_folds_report.csv"
    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(report_csv_path, index=False, encoding="utf-8")
    print(f"[+] Đã lưu: {report_csv_path}")

    # 6. Tóm tắt
    print("\n" + "=" * 60)
    print("TÓM TẮT FOLD DISTRIBUTION")
    print("=" * 60)
    for fold_idx in range(3):
        val_count = int((df["fold"] == fold_idx).sum())
        train_count = n - val_count
        print(f"  Fold {fold_idx}: Train={train_count}, Val={val_count}")

    # Kiểm tra OUT_OF_SCOPE tất cả fold
    print("\n  Kiểm tra OUT_OF_SCOPE:")
    for fold_idx in range(3):
        val_oos = "OUT_OF_SCOPE" in df[df["fold"] == fold_idx]["intent"].values
        train_oos = "OUT_OF_SCOPE" in df[df["fold"] != fold_idx]["intent"].values
        status = "[OK]" if val_oos else "[CẢNH BÁO]"
        print(f"    Fold {fold_idx}: Val OUT_OF_SCOPE={val_oos} {status} | Train OUT_OF_SCOPE={train_oos}")


if __name__ == "__main__":
    main()

