r"""Phase 3F: Dự đoán 28 câu Intent test bằng mô hình TextCNN cuối.

Đặt tại: Edge_AI/src/intent/predict_final.py
Chạy từ project root (PowerShell):
    .\.venv\Scripts\python.exe src\intent\predict_final.py --check-only
    .\.venv\Scripts\python.exe src\intent\predict_final.py --predict

Không train, không sửa test, không ghi đè kết quả hiện có, không tạo submission.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.intent.preprocess import IntentPreprocessor, LABEL_TO_ID

TEST_PATH = PROJECT_ROOT / "data" / "raw" / "intent_test.csv"
SAMPLE_PATH = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "intent_final" / "final.keras"
PREP_PATH = PROJECT_ROOT / "models" / "intent_final" / "final_preprocessor.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions"
PRED_PATH = OUTPUT_DIR / "intent_predictions.csv"
DETAIL_PATH = OUTPUT_DIR / "intent_confidence.csv"


def inspect_inputs():
    """Kiểm tra dữ liệu và yêu cầu submission, không suy đoán nhãn test."""
    for path in (TEST_PATH, SAMPLE_PATH, MODEL_PATH, PREP_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu file: {path}")

    df_test = pd.read_csv(TEST_PATH, encoding="utf-8")
    sample = pd.read_csv(SAMPLE_PATH, encoding="utf-8", keep_default_na=False)

    if list(df_test.columns) != ["question_id", "question"]:
        raise ValueError(f"intent_test.csv sai cột: {list(df_test.columns)}")
    if list(sample.columns) != ["task_type", "sample_id", "prediction"]:
        raise ValueError(f"sample_submission.csv sai cột: {list(sample.columns)}")
    if len(df_test) != 28:
        raise ValueError(f"Số câu test phải là 28, thực tế: {len(df_test)}")
    if df_test.isna().any().any() or df_test["question"].astype(str).str.strip().eq("").any():
        raise ValueError("Có question_id hoặc câu hỏi trống")
    if df_test["question_id"].duplicated().any():
        raise ValueError("Có question_id trùng lặp")
    if sample["sample_id"].duplicated().any():
        raise ValueError("sample_submission.csv có sample_id trùng")

    intent_sample = sample[sample["task_type"] == "INTENT"].copy()
    detection_sample = sample[sample["task_type"] == "DETECTION"]
    if len(intent_sample) != 28 or len(detection_sample) != 60 or len(sample) != 88:
        raise ValueError("sample_submission.csv không có đúng 60 DETECTION + 28 INTENT")
    actual_ids = set(df_test["question_id"].astype(str))
    expected_ids = set(intent_sample["sample_id"].astype(str))
    if actual_ids != expected_ids:
        raise ValueError(
            "question_id không khớp sample_submission.csv; "
            f"thiếu: {sorted(expected_ids - actual_ids)}, dư: {sorted(actual_ids - expected_ids)}"
        )

    # Sắp đúng thứ tự 28 sample_id của file mẫu để tránh thay đổi thứ tự submission sau này.
    ordered_test = df_test.set_index("question_id").loc[intent_sample["sample_id"].tolist()].reset_index()
    print("=== PHASE 3F: KIỂM TRA INTENT INFERENCE ===")
    print(f"Test Intent: {len(ordered_test)} câu")
    print(f"Submission mẫu: {len(detection_sample)} DETECTION + {len(intent_sample)} INTENT")
    print("ID không thiếu, không trùng, đúng thứ tự file mẫu: OK")
    print(f"Model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Preprocessor: {PREP_PATH.relative_to(PROJECT_ROOT)}")
    return ordered_test


def predict(df):
    if PRED_PATH.exists() or DETAIL_PATH.exists():
        raise FileExistsError(
            "Đã tồn tại kết quả dự đoán, không ghi đè: "
            + ", ".join(str(p) for p in (PRED_PATH, DETAIL_PATH) if p.exists())
        )

    from tensorflow import keras

    prep = IntentPreprocessor.load(PREP_PATH)
    model = keras.models.load_model(MODEL_PATH, compile=False)
    texts = df["question"].astype(str).tolist()
    X = prep.transform_texts(texts)
    probs = np.asarray(model.predict(X, verbose=0))

    if probs.shape != (len(df), 5):
        raise ValueError(f"Sai output shape: {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model dự đoán NaN hoặc Inf")
    if (probs < -1e-6).any() or (probs > 1 + 1e-6).any():
        raise ValueError("Xác suất ngoài khoảng [0, 1]")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-4):
        raise ValueError("Tổng xác suất các lớp khác 1")

    class_ids = np.argmax(probs, axis=1)
    predictions = prep.decode_labels(class_ids)
    if not set(predictions).issubset(LABEL_TO_ID.keys()):
        raise ValueError("Model trả nhãn không hợp lệ")

    results = pd.DataFrame({
        "sample_id": df["question_id"].astype(str).tolist(),
        "prediction": predictions,
    })
    details = pd.DataFrame({
        "sample_id": df["question_id"].astype(str).tolist(),
        "prediction": predictions,
        "confidence": np.max(probs, axis=1).round(6),
    })
    # CSV chỉ có dự đoán model, không tạo nhãn test và không ghi ra file submission cuối.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(PRED_PATH, index=False, encoding="utf-8")
    details.to_csv(DETAIL_PATH, index=False, encoding="utf-8")

    print("\nINFERENCE COMPLETED")
    print("Số dự đoán:", len(results))
    print("Phân bố nhãn dự đoán:")
    print(results["prediction"].value_counts().to_string())
    print("Đã lưu:", PRED_PATH)
    print("Đã lưu:", DETAIL_PATH)
    print("Chưa nộp Kaggle, chưa tạo submission.csv")


def main():
    parser = argparse.ArgumentParser(description="Inference Intent TextCNN final")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true", help="Kiểm tra file và ID, chưa predict")
    group.add_argument("--predict", action="store_true", help="Chạy model predict cho 28 câu test")
    args = parser.parse_args()
    df = inspect_inputs()
    if args.check_only:
        print("CHECK ONLY OK: chưa chạy model và chưa ghi file nào.")
        return
    predict(df)


if __name__ == "__main__":
    main()
