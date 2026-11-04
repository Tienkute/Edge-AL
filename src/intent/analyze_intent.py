"""
Phân tích và khám phá dữ liệu Intent Classification (EDA)
EDGE AI CHALLENGE 2026 – Warm-up
"""

import csv
from pathlib import Path
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thiết lập đường dẫn gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def analyze_intent():
    data_raw = PROJECT_ROOT / "data" / "raw"
    intent_train_path = data_raw / "intent_train.csv"
    intent_test_path = data_raw / "intent_test.csv"

    output_eda_dir = PROJECT_ROOT / "outputs" / "eda"
    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    output_eda_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHÂN TÍCH TẬP DỮ LIỆU INTENT CLASSIFICATION")
    print("=" * 60)

    # 1. Đọc dữ liệu train
    df_train = pd.read_csv(intent_train_path)
    train_samples = len(df_train)
    print(f"\n[+] Số mẫu huấn luyện (Train samples): {train_samples}")

    # A. Thống kê Intent
    intent_counts = df_train["intent"].value_counts().sort_index()
    intent_dist_list = []
    print("\n--- A. THỐNG KÊ PHÂN BỐ INTENT ---")
    print(f"{'Intent':<18} {'Samples':<10} {'Percentage (%)':<15}")
    print("-" * 45)
    for intent, count in intent_counts.items():
        pct = (count / train_samples) * 100
        intent_dist_list.append((intent, count, round(pct, 2)))
        print(f"{intent:<18} {count:<10} {pct:>6.2f}%")

    # Lưu outputs/eda/intent_distribution.csv
    intent_dist_csv = output_eda_dir / "intent_distribution.csv"
    with open(intent_dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["intent", "sample_count", "percentage"])
        writer.writerows(intent_dist_list)
    print(f"\n[+] Đã lưu: {intent_dist_csv}")

    # B. Phân tích độ dài câu (word_count)
    df_train["word_count"] = df_train["question"].apply(lambda q: len(str(q).split()))
    w_min = int(df_train["word_count"].min())
    w_max = int(df_train["word_count"].max())
    w_mean = float(df_train["word_count"].mean())
    w_median = float(df_train["word_count"].median())

    print("\n--- B. ĐỘ DÀI CÂU HỎI (WORD COUNT) ---")
    print(f"min:    {w_min}")
    print(f"max:    {w_max}")
    print(f"mean:   {w_mean:.2f}")
    print(f"median: {w_median:.1f}")

    # Lưu outputs/eda/intent_question_lengths.csv
    intent_lens_csv = output_eda_dir / "intent_question_lengths.csv"
    df_lengths = df_train[["question_id", "intent", "word_count"]]
    df_lengths.to_csv(intent_lens_csv, index=False, encoding="utf-8")
    print(f"[+] Đã lưu: {intent_lens_csv}")

    # C. Vocabulary
    # Đơn giản: lowercase và split theo khoảng trắng, không stemming, không tokenizer phức tạp
    all_tokens = []
    for q in df_train["question"]:
        words = str(q).lower().split()
        all_tokens.extend(words)

    total_tokens = len(all_tokens)
    vocab_counter = Counter(all_tokens)
    vocab_size = len(vocab_counter)
    top_10_words = vocab_counter.most_common(10)

    print("\n--- C. TỪ VỰNG (VOCABULARY) ---")
    print(f"Total tokens:           {total_tokens}")
    print(f"Unique vocabulary size: {vocab_size}")
    print("\n10 từ xuất hiện nhiều nhất:")
    for rank, (word, count) in enumerate(top_10_words, start=1):
        print(f"  {rank:>2}. '{word}': {count} lần")

    # 6. Vẽ biểu đồ Intent (Không dùng seaborn, không chỉ định màu thủ công)
    print("\n[+] Đang vẽ biểu đồ Intent...")

    # Biểu đồ 1: Bar chart intent_distribution.png
    plt.figure(figsize=(8, 5))
    plt.bar(intent_counts.index, intent_counts.values)
    plt.xlabel("Intent")
    plt.ylabel("Number of Samples")
    plt.title("Intent Distribution (Train Set)")
    plt.tight_layout()
    fig1_path = output_fig_dir / "intent_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu: {fig1_path}")

    # Biểu đồ 2: Histogram intent_length_distribution.png
    plt.figure(figsize=(8, 5))
    plt.hist(df_train["word_count"], bins=range(w_min, w_max + 2), align="left", rwidth=0.8)
    plt.xlabel("Word Count")
    plt.ylabel("Number of Questions")
    plt.title("Question Length Distribution (Word Count)")
    plt.tight_layout()
    fig2_path = output_fig_dir / "intent_length_distribution.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu: {fig2_path}")

    # 7. Kiểm tra test Intent (KHÔNG DỰ ĐOÁN, KHÔNG SUY ĐOÁN LABEL)
    print("\n--- D. KIỂM TRA TẬP TEST INTENT (READ-ONLY) ---")
    df_test = pd.read_csv(intent_test_path)
    test_samples = len(df_test)
    test_dup_ids = int(df_test["question_id"].duplicated().sum())
    test_missing_q = int(df_test["question"].isna().sum())
    test_word_counts = df_test["question"].apply(lambda q: len(str(q).split()))
    t_min = int(test_word_counts.min())
    t_max = int(test_word_counts.max())
    t_mean = float(test_word_counts.mean())

    print(f"Number of rows:             {test_samples}")
    print(f"question_id duplicates:     {test_dup_ids}")
    print(f"missing question rows:      {test_missing_q}")
    print(f"word_count min/max/mean:    {t_min} / {t_max} / {t_mean:.2f}")

    # 8. Cập nhật outputs/eda/eda_summary.txt với phần [INTENT]
    eda_summary_path = output_eda_dir / "eda_summary.txt"

    # Đọc nội dung hiện có (nếu analyze_dataset đã chạy)
    existing_content = ""
    if eda_summary_path.exists():
        with open(eda_summary_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

    intent_section_lines = [
        "[INTENT]",
        "",
        f"Train samples: {train_samples}",
        "",
        "Intent distribution:",
    ]
    for intent, count, pct in intent_dist_list:
        intent_section_lines.append(f"  {intent}: {count} ({pct}%)")

    intent_section_lines.extend([
        "",
        "Question length:",
        f"min: {w_min}",
        f"max: {w_max}",
        f"mean: {w_mean:.2f}",
        f"median: {w_median:.1f}",
        "",
        "Vocabulary size:",
        f"total tokens: {total_tokens}",
        f"unique vocabulary size: {vocab_size}",
        "",
        f"Intent test samples: {test_samples}",
    ])
    intent_text = "\n".join(intent_section_lines)

    if "[INTENT]" in existing_content:
        # Thay thế phần [INTENT] cũ
        parts = existing_content.split("[INTENT]")
        final_summary = parts[0] + intent_text + "\n"
    elif "[DETECTION]" in existing_content:
        # Ghép thêm phần [INTENT]
        final_summary = existing_content.strip() + "\n\n" + intent_text + "\n"
    else:
        # Chưa có phần detection, tạo mới
        final_summary = "=== EDGE AI CHALLENGE 2026 - EDA SUMMARY ===\n\n" + intent_text + "\n"

    with open(eda_summary_path, "w", encoding="utf-8") as f:
        f.write(final_summary)
    print(f"\n[+] Đã cập nhật file báo cáo tổng hợp: {eda_summary_path}")

    return {
        "train_samples": train_samples,
        "intent_dist": intent_dist_list,
        "word_count_min": w_min,
        "word_count_max": w_max,
        "word_count_mean": w_mean,
        "word_count_median": w_median,
        "total_tokens": total_tokens,
        "vocab_size": vocab_size,
        "top_10_words": top_10_words,
        "test_samples": test_samples,
    }


if __name__ == "__main__":
    analyze_intent()

