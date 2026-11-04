r"""Phase 3E: Huấn luyện TextCNN cuối cùng trên 63 câu TRAIN (không dùng Kaggle test).

Đặt file tại Edge_AI/src/intent/train_final.py.
Chạy tại thư mục gốc Edge_AI:
    .\.venv\Scripts\python.exe src\intent\train_final.py --check-only
    .\.venv\Scripts\python.exe src\intent\train_final.py --train

Giữ nguyên mô hình baseline và weighted CV; không ghi đè kết quả cũ.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intent.preprocess import IntentPreprocessor, LABEL_TO_ID
from src.intent.model import build_textcnn_from_config, load_config

MODEL_DIR = ROOT / "models" / "intent_final"
OUTPUT_DIR = ROOT / "outputs" / "intent_final"
DATA_PATH = ROOT / "data" / "raw" / "intent_train.csv"
CONFIG_PATH = ROOT / "configs" / "intent.yaml"
LABELS = list(LABEL_TO_ID)


def validate_data():
    cfg = load_config(CONFIG_PATH)
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    required = {"question_id", "question", "intent"}
    if not required.issubset(df.columns):
        raise ValueError(f"Thiếu cột dữ liệu: {required - set(df.columns)}")
    if len(df) != 63:
        raise ValueError(f"Mong đợi 63 câu train, thực tế {len(df)}")
    if df[["question_id", "question", "intent"]].isna().any().any():
        raise ValueError("Có ô dữ liệu train bị trống")
    if df.question_id.duplicated().any():
        raise ValueError("Có question_id trùng trong train")
    if set(df.intent) != set(LABELS):
        raise ValueError(f"Sai bộ nhãn train: {set(df.intent)}")
    if not df.question.astype(str).str.strip().all():
        raise ValueError("Có câu hỏi rỗng")
    return cfg, df


def compute_weights(y):
    counts = np.bincount(y, minlength=len(LABELS))
    if np.any(counts == 0):
        raise ValueError("Có lớp thiếu mẫu huấn luyện")
    n = len(y)
    return {int(i): float(n / (len(LABELS) * count)) for i, count in enumerate(counts)}


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện TextCNN cuối trên 63 câu")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra cấu hình và train data")
    group.add_argument("--train", action="store_true", help="Train và lưu mô hình cuối")
    args = parser.parse_args()

    cfg, df = validate_data()
    seed = int(cfg.get("seed", 42))
    max_length = int(cfg.get("max_length", 24))
    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 8))
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs và batch_size phải là số nguyên dương")

    prep = IntentPreprocessor(max_length=max_length)
    texts = df.question.astype(str).tolist()
    labels = df.intent.tolist()
    prep.fit(texts)  # Chỉ fit trên toàn bộ TRAIN (không dùng TEST Kaggle)
    X, y = prep.transform(texts, labels)
    weights = compute_weights(y)

    print("\n=== PHASE 3E: FINAL TEXTCNN ===")
    print(f"Train: {len(df)} câu | X shape: {X.shape} | vocab: {prep.vocab_size}")
    print(f"epochs: {epochs} | batch_size: {batch_size} | seed: {seed}")
    for label, class_id in LABEL_TO_ID.items():
        n = int(np.sum(y == class_id))
        print(f"{label:13s}: {n:2d} câu, class_weight = {weights[class_id]:.4f}")

    if args.check_only:
        print("CHECK ONLY OK: chưa huấn luyện, chưa ghi file nào.")
        return

    model_path = MODEL_DIR / "final.keras"
    prep_path = MODEL_DIR / "final_preprocessor.json"
    meta_path = OUTPUT_DIR / "training_summary.json"
    history_path = OUTPUT_DIR / "training_history.csv"
    existing = [p for p in (model_path, prep_path, meta_path, history_path) if p.exists()]
    if existing:
        raise FileExistsError(
            "Đã có kết quả final; dừng để tránh ghi đè: " + ", ".join(str(p) for p in existing)
        )

    import tensorflow as tf
    from tensorflow import keras

    keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    model = build_textcnn_from_config(cfg, vocab_size=prep.vocab_size, compile_model=True)
    print("\nĐang train mô hình cuối trên 63 câu. KHÔNG sử dụng validation/test.")
    # Epochs=30 lấy từ CV trước đó: cả 3 fold đều đạt best_epoch=30.
    # Không dùng EarlyStopping val_loss khi đã chọn train trên toàn bộ 63 câu.
    history = model.fit(
        X, y, epochs=epochs, batch_size=batch_size,
        class_weight=weights, verbose=1, shuffle=True,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    prep.save(prep_path)
    pd.DataFrame(history.history).to_csv(history_path, index=False, encoding="utf-8")

    # Kiểm tra khả năng nạp lại và kết quả forward pass giống nhau.
    check_x = X[: min(8, len(X))]
    original_probs = model.predict(check_x, verbose=0)
    loaded_model = keras.models.load_model(model_path)
    loaded_prep = IntentPreprocessor.load(prep_path)
    reloaded_x = loaded_prep.transform_texts(texts[: len(check_x)])
    loaded_probs = loaded_model.predict(reloaded_x, verbose=0)
    if not np.array_equal(check_x, reloaded_x):
        raise AssertionError("Preprocessor sau khi nạp không tạo cùng input")
    if not np.allclose(original_probs, loaded_probs, rtol=1e-5, atol=1e-5):
        raise AssertionError("Model sau khi nạp cho xác suất khác model gốc")

    summary = {
        "purpose": "Final Intent TextCNN trained on full labeled train set",
        "train_samples": len(df),
        "train_only": True,
        "cross_validation_oof_accuracy_prior_experiment": 0.9682539682539683,
        "note": "OOF score thuộc weighted CV trước đó, KHÔNG phải điểm validation của final model.",
        "epochs_trained": len(history.history["loss"]),
        "vocab_size": prep.vocab_size,
        "num_parameters": model.count_params(),
        "class_weights": {label: weights[cid] for label, cid in LABEL_TO_ID.items()},
        "saved_model": str(model_path.relative_to(ROOT)),
        "saved_preprocessor": str(prep_path.relative_to(ROOT)),
        "reload_check": "PASSED",
    }
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nFINAL TRAIN COMPLETED")
    print("Model:", model_path)
    print("Preprocessor:", prep_path)
    print("Training history:", history_path)
    print("Summary:", meta_path)
    print("Kiểm tra save/load: PASSED")
    print("Chưa dự đoán intent_test.csv, chưa tạo submission.csv.")


if __name__ == "__main__":
    main()
