r"""Phase 3D: Thử Class Weight cho TextCNN, không ghi đè baseline.

Chạy từ thư mục gốc Edge_AI (PowerShell):
    .\.venv\Scripts\python.exe src\intent\train_weighted.py --check-only
    .\.venv\Scripts\python.exe src\intent\train_weighted.py --train

Mỗi fold được train lại từ đầu, vocabulary chỉ fit trên train fold.
Dữ liệu test Kaggle KHÔNG được sử dụng.
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

LABELS = list(LABEL_TO_ID.keys())
MODEL_DIR = ROOT / "models" / "intent_weighted"
OUT_DIR = ROOT / "outputs" / "intent_weighted"


def read_and_validate():
    """Đọc đúng dữ liệu TRAIN, kiểm tra manifest folds; không đụng đến test."""
    cfg = load_config(ROOT / "configs" / "intent.yaml")
    train = pd.read_csv(ROOT / "data" / "raw" / "intent_train.csv")
    folds = pd.read_csv(ROOT / "data" / "processed" / "intent_folds.csv")

    required_train = {"question_id", "question", "intent"}
    required_folds = {"question_id", "fold"}
    if not required_train.issubset(train.columns) or not required_folds.issubset(folds.columns):
        raise ValueError("Thiếu cột bắt buộc trong intent_train.csv hoặc intent_folds.csv")
    if len(train) != 63 or len(folds) != 63:
        raise ValueError(f"Số mẫu không đúng: train={len(train)}, folds={len(folds)} (mong đợi 63)")
    if train.question_id.isna().any() or folds.question_id.isna().any():
        raise ValueError("Có question_id bị trống")
    if train.question_id.duplicated().any() or folds.question_id.duplicated().any():
        raise ValueError("Có question_id bị trùng")
    if set(train.question_id) != set(folds.question_id):
        raise ValueError("Danh sách question_id giữa train và folds không khớp")
    if train.question.isna().any() or train.intent.isna().any():
        raise ValueError("Có câu hỏi hoặc nhãn bị trống")
    if set(train.intent) != set(LABELS):
        raise ValueError(f"Nhãn train không khớp 5 nhãn bắt buộc: {set(train.intent)}")
    if set(folds.fold) != {0, 1, 2}:
        raise ValueError("Fold phải có đúng các giá trị 0, 1, 2")
    data = train.merge(folds, on="question_id", validate="one_to_one")
    for fold in range(3):
        tr = data[data.fold != fold]
        va = data[data.fold == fold]
        if len(tr) != 42 or len(va) != 21:
            raise ValueError(f"Fold {fold}: train={len(tr)}, val={len(va)} (mong đợi 42/21)")
        if (tr.intent == "OUT_OF_SCOPE").sum() != 2 or (va.intent == "OUT_OF_SCOPE").sum() != 1:
            raise ValueError(f"Fold {fold}: phân bố OUT_OF_SCOPE không đúng")
        if set(tr.intent) != set(LABELS):
            raise ValueError(f"Fold {fold}: tập train thiếu class")
    return cfg, data


def balanced_weights(y_train):
    """w_c = N / (K * n_c), chỉ tính từ nhãn TRAIN fold hiện tại."""
    n_total = len(y_train)
    n_classes = len(LABELS)
    counts = np.bincount(y_train, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError("Train fold đang thiếu ít nhất một lớp")
    return {int(i): float(n_total / (n_classes * count)) for i, count in enumerate(counts)}


def show_plan(cfg, data):
    print("\nPHASE 3D — KIỂM TRA TRƯỚC KHI TRAIN")
    print(f"Số câu train: {len(data)}; folds: 3")
    print(f"Epochs: {cfg.get('epochs', 30)} | batch_size: {cfg.get('batch_size', 8)}")
    for fold in range(3):
        tr = data[data.fold != fold]
        y = np.array([LABEL_TO_ID[s] for s in tr.intent], dtype=np.int32)
        weights = balanced_weights(y)
        parts = ", ".join(f"{label}={weights[i]:.3f}" for label, i in LABEL_TO_ID.items())
        print(f"Fold {fold}: train=42, val=21 | Class Weights: {parts}")
    print("Baseline sẽ được giữ nguyên: models/intent/ và outputs/intent_cv/")


def train_weighted(cfg, data):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    # Không vô tình ghi đè kết quả Weighted đã tồn tại từ lần chạy trước.
    expected_files = [MODEL_DIR / f"fold_{i}.keras" for i in range(3)]
    expected_files += [OUT_DIR / "cv_summary.json", OUT_DIR / "oof_predictions.csv"]
    if any(path.exists() for path in expected_files):
        raise FileExistsError(
            "Đã có kết quả trong models/intent_weighted hoặc outputs/intent_weighted. "
            "Không tự động ghi đè. Hãy kiểm tra file cũ trước khi chạy lại."
        )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    max_length = int(cfg.get("max_length", 24))
    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 8))

    oof_rows = []
    fold_rows = []
    class_weight_report = {}

    for fold in range(3):
        keras.backend.clear_session()
        random.seed(seed + fold * 100)
        np.random.seed(seed + fold * 100)
        tf.keras.utils.set_random_seed(seed + fold * 100)

        tr = data[data.fold != fold].reset_index(drop=True)
        va = data[data.fold == fold].reset_index(drop=True)
        prep = IntentPreprocessor(max_length=max_length)
        prep.fit(tr.question.tolist())  # Không fit trên validation!
        X_train, y_train = prep.transform(tr.question.tolist(), tr.intent.tolist())
        X_val, y_val = prep.transform(va.question.tolist(), va.intent.tolist())

        weights = balanced_weights(y_train)
        class_weight_report[f"fold_{fold}"] = {label: weights[i] for label, i in LABEL_TO_ID.items()}
        model = build_textcnn_from_config(cfg, vocab_size=prep.vocab_size, compile_model=True)
        stopper = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, mode="min", restore_best_weights=True
        )
        print(f"\n--- TRAIN WEIGHTED FOLD {fold}: 42 train / 21 val ---")
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=weights,
            callbacks=[stopper],
            verbose=1,
        )

        probs = model.predict(X_val, verbose=0)
        preds = np.argmax(probs, axis=1)
        true_labels = va.intent.tolist()
        pred_labels = prep.decode_labels(preds)
        acc = accuracy_score(true_labels, pred_labels)
        macro = f1_score(true_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
        best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"Fold {fold}: accuracy={acc:.4f}, macro-F1={macro:.4f}, best epoch={best_epoch}")

        model_path = MODEL_DIR / f"fold_{fold}.keras"
        prep_path = MODEL_DIR / f"fold_{fold}_preprocessor.json"
        model.save(model_path)
        prep.save(prep_path)
        # Xác minh model và vocabulary sau lưu/nạp.
        reloaded_model = keras.models.load_model(model_path)
        reloaded_prep = IntentPreprocessor.load(prep_path)
        X_reloaded = reloaded_prep.transform_texts(va.question.tolist())
        probs_reloaded = reloaded_model.predict(X_reloaded, verbose=0)
        if not np.allclose(probs, probs_reloaded, atol=1e-5, rtol=1e-5):
            raise AssertionError(f"Fold {fold}: xác suất khác sau khi load model/preprocessor")

        fold_rows.append({
            "fold": fold,
            "train_samples": len(tr),
            "val_samples": len(va),
            "best_epoch": best_epoch,
            "epochs_ran": len(hist.history["loss"]),
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "val_loss": float(min(hist.history["val_loss"])),
            "vocab_size": int(prep.vocab_size),
        })
        for i, row in va.iterrows():
            oof_rows.append({
                "question_id": row.question_id,
                "fold": fold,
                "true_intent": row.intent,
                "predicted_intent": pred_labels[i],
                "confidence": float(np.max(probs[i])),
            })

    oof = pd.DataFrame(oof_rows)
    if len(oof) != 63 or oof.question_id.duplicated().any():
        raise AssertionError("OOF không đủ 63 câu riêng biệt")
    overall_acc = accuracy_score(oof.true_intent, oof.predicted_intent)
    overall_f1 = f1_score(
        oof.true_intent, oof.predicted_intent, labels=LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(oof.true_intent, oof.predicted_intent, labels=LABELS)
    report = classification_report(
        oof.true_intent, oof.predicted_intent, labels=LABELS, zero_division=0
    )
    per_class = {
        label: {"total": int((oof.true_intent == label).sum()),
                "correct": int(((oof.true_intent == label) & (oof.predicted_intent == label)).sum())}
        for label in LABELS
    }
    summary = {
        "experiment": "phase3d_balanced_class_weight_only",
        "notes": "Chỉ thay class_weight; giữ nguyên config/manifest của baseline.",
        "folds": fold_rows,
        "overall_oof_accuracy": float(overall_acc),
        "overall_oof_macro_f1": float(overall_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "class_weights_by_fold": class_weight_report,
    }
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False, encoding="utf-8")
    oof[oof.true_intent != oof.predicted_intent].to_csv(
        OUT_DIR / "misclassified_samples.csv", index=False, encoding="utf-8"
    )
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    (OUT_DIR / "cv_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== KẾT QUẢ THỬ NGHIỆM CLASS WEIGHT ===")
    print(f"OOF Accuracy: {overall_acc:.4f} | OOF Macro-F1: {overall_f1:.4f}")
    print("Số câu đúng từng lớp:", per_class)
    print(report)
    baseline_path = ROOT / "outputs" / "intent_cv" / "cv_summary.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        print("=== SO SÁNH VỚI BASELINE ===")
        print(f"Accuracy: {baseline['overall_oof_accuracy']:.4f} → {overall_acc:.4f}")
        print(f"Macro-F1: {baseline['overall_oof_macro_f1']:.4f} → {overall_f1:.4f}")
    print(f"Đã lưu model tại: {MODEL_DIR}")
    print(f"Đã lưu kết quả tại: {OUT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3D - kiểm tra/train TextCNN có Class Weight")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check-only", action="store_true", help="Chỉ kiểm tra fold/weights, KHÔNG train")
    group.add_argument("--train", action="store_true", help="Thực sự train 3 folds và lưu model riêng")
    args = parser.parse_args()
    cfg, data = read_and_validate()
    show_plan(cfg, data)
    if args.train:
        train_weighted(cfg, data)
    else:
        print("\nCHECK ONLY: hoàn tất, chưa huấn luyện, không ghi file nào.")


if __name__ == "__main__":
    main()
