"""
Huấn luyện TextCNN cho bài toán Intent Classification bằng 3-Fold Cross-Validation.
EDGE AI CHALLENGE 2026 – Warm-up

Quy trình:
  1. Đọc dữ liệu từ data/raw/intent_train.csv và phân chia fold từ data/processed/intent_folds.csv.
  2. Với mỗi fold (0, 1, 2):
     a. Tách train/val.
     b. Fit vocabulary CHỈ trên tập train của fold đó (chống data leakage).
     c. Khởi tạo mô hình TextCNN mới hoàn toàn (reset session và weights).
     d. Huấn luyện với EarlyStopping (monitor='val_loss', patience=6, restore_best_weights=True).
     e. Đánh giá trên validation fold (Accuracy, Macro-F1).
     f. Lưu model vào models/intent/fold_{fold}.keras và preprocessor vào models/intent/fold_{fold}_preprocessor.json.
     g. Lưu biểu đồ learning curves vào outputs/figures/.
  3. Tổng hợp Out-of-Fold (OOF) predictions:
     - Lưu outputs/intent_cv/oof_predictions.csv (đủ 63 câu).
     - Lưu outputs/intent_cv/misclassified_samples.csv (các câu đoán sai).
     - Tính Overall OOF Accuracy, Macro-F1.
     - Vẽ và lưu Confusion Matrix: outputs/intent_cv/oof_confusion_matrix.png.
     - Lưu Classification Report: outputs/intent_cv/oof_classification_report.txt.
     - Lưu tóm tắt kết quả: outputs/intent_cv/cv_summary.json.
  4. Kiểm tra (load test) các mô hình .keras và preprocessor đã lưu để đảm bảo khả năng tái tạo.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

from src.intent.evaluate import (
    INTENT_CLASSES,
    compute_metrics,
    plot_confusion_matrix,
    plot_learning_curves,
)
from src.intent.model import build_textcnn_from_config, load_config
from src.intent.preprocess import IntentPreprocessor


def set_seed(seed: int = 42) -> None:
    """Thiết lập random seed cố định để đảm bảo tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_3fold_cross_validation() -> Dict[str, Any]:
    print("=" * 65)
    print("PHASE 3C: HUẤN LUYỆN TEXTCNN VỚI 3-FOLD CROSS-VALIDATION")
    print("=" * 65)

    data_raw = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models" / "intent"
    output_cv_dir = PROJECT_ROOT / "outputs" / "intent_cv"
    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    config_path = PROJECT_ROOT / "configs" / "intent.yaml"

    models_dir.mkdir(parents=True, exist_ok=True)
    output_cv_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Đọc dữ liệu
    print("\n[1] Đọc dữ liệu và cấu hình...")
    intent_train_csv = data_raw / "intent_train.csv"
    intent_folds_csv = processed_dir / "intent_folds.csv"

    if not intent_train_csv.exists() or not intent_folds_csv.exists():
        raise FileNotFoundError("Thiếu intent_train.csv hoặc intent_folds.csv!")

    df_train_all = pd.read_csv(intent_train_csv)
    df_folds = pd.read_csv(intent_folds_csv)

    df_merged = df_train_all.merge(df_folds, on="question_id", how="inner")
    total_samples = len(df_merged)
    print(f"    Tổng số câu hỏi huấn luyện: {total_samples}")

    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 8))
    max_epochs = int(cfg.get("epochs", 30))
    max_length = int(cfg.get("max_length", 24))

    print(f"    Cấu hình: max_length={max_length}, batch_size={batch_size}, max_epochs={max_epochs}, seed={seed}")

    fold_results = []
    oof_records = []
    fold_histories = {}

    # 2. Vòng lặp huấn luyện 3 Folds
    for fold in range(3):
        print("\n" + "-" * 65)
        print(f"HUẤN LUYỆN FOLD {fold}")
        print("-" * 65)

        # Đặt lại seed cho mỗi fold để đảm bảo tính xác định
        set_seed(seed + fold * 100)
        keras.backend.clear_session()

        # Tách dữ liệu fold
        df_train_fold = df_merged[df_merged["fold"] != fold].reset_index(drop=True)
        df_val_fold = df_merged[df_merged["fold"] == fold].reset_index(drop=True)

        n_train = len(df_train_fold)
        n_val = len(df_val_fold)
        print(f"  Train: {n_train} mẫu | Val: {n_val} mẫu")

        train_texts = df_train_fold["question"].tolist()
        train_labels = df_train_fold["intent"].tolist()
        val_texts = df_val_fold["question"].tolist()
        val_labels = df_val_fold["intent"].tolist()
        val_qids = df_val_fold["question_id"].tolist()

        # Fit preprocessor CHỈ trên tập train của fold
        preprocessor = IntentPreprocessor(max_length=max_length)
        preprocessor.fit(train_texts)
        vocab_size = preprocessor.vocab_size
        print(f"  Vocabulary size (Fold {fold}): {vocab_size} tokens")

        # Transform train và val
        X_train, y_train = preprocessor.transform(train_texts, train_labels)
        X_val, y_val = preprocessor.transform(val_texts, val_labels)

        # Khởi tạo mô hình mới hoàn toàn
        model = build_textcnn_from_config(cfg, vocab_size=vocab_size, compile_model=True)

        # Thiết lập EarlyStopping
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            mode="min",
            verbose=1,
        )

        # Huấn luyện
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1,
        )

        hist = history.history
        fold_histories[fold] = hist

        # Tìm best epoch dựa trên val_loss nhỏ nhất
        val_losses = hist["val_loss"]
        best_epoch_idx = int(np.argmin(val_losses))
        best_epoch = best_epoch_idx + 1
        best_val_loss = float(val_losses[best_epoch_idx])
        best_val_acc_hist = float(hist["val_accuracy"][best_epoch_idx])

        # Dự đoán trên tập validation sau khi đã restore_best_weights
        val_probs = model.predict(X_val, verbose=0)
        val_pred_ids = np.argmax(val_probs, axis=-1)
        val_confidences = np.max(val_probs, axis=-1)
        val_pred_labels = preprocessor.decode_labels(val_pred_ids)

        # Tính metrics cho fold
        fold_eval = compute_metrics(val_labels, val_pred_labels, labels=INTENT_CLASSES)
        fold_acc = fold_eval["accuracy"]
        fold_f1 = fold_eval["macro_f1"]

        print(f"\n  [+] Kết quả Fold {fold}:")
        print(f"      Best Epoch:      {best_epoch}/{len(val_losses)}")
        print(f"      Best Val Loss:   {best_val_loss:.4f}")
        print(f"      Val Accuracy:    {fold_acc:.4f} ({fold_acc * 100:.2f}%)")
        print(f"      Val Macro-F1:    {fold_f1:.4f}")

        # Lưu model và preprocessor
        model_save_path = models_dir / f"fold_{fold}.keras"
        prep_save_path = models_dir / f"fold_{fold}_preprocessor.json"

        model.save(model_save_path)
        preprocessor.save(prep_save_path)
        print(f"      Đã lưu model:         {model_save_path}")
        print(f"      Đã lưu preprocessor:  {prep_save_path}")

        # Vẽ biểu đồ learning curves
        fig_path = output_fig_dir / f"intent_fold_{fold}_learning_curves.png"
        plot_learning_curves(hist, fig_path, title_prefix=f"TextCNN Fold {fold}")
        print(f"      Đã lưu learning curve: {fig_path}")

        # Lưu kết quả fold
        fold_results.append({
            "fold": fold,
            "train_samples": n_train,
            "val_samples": n_val,
            "best_epoch": best_epoch,
            "total_epochs_trained": len(val_losses),
            "val_loss": round(best_val_loss, 4),
            "accuracy": round(fold_acc, 4),
            "macro_f1": round(fold_f1, 4),
            "vocab_size": vocab_size,
        })

        # Lưu OOF records
        for i in range(n_val):
            oof_records.append({
                "question_id": val_qids[i],
                "question": val_texts[i],
                "fold": fold,
                "true_intent": val_labels[i],
                "predicted_intent": val_pred_labels[i],
                "confidence": round(float(val_confidences[i]), 4),
            })

    # 3. Tổng hợp Out-of-Fold Predictions
    print("\n" + "=" * 65)
    print("TỔNG HỢP KẾT QUẢ OUT-OF-FOLD (OOF)")
    print("=" * 65)

    df_oof_full = pd.DataFrame(oof_records)
    # Lưu file oof_predictions.csv (không chứa cột question theo yêu cầu đề bài)
    oof_csv_path = output_cv_dir / "oof_predictions.csv"
    df_oof_export = df_oof_full[["question_id", "fold", "true_intent", "predicted_intent", "confidence"]]
    df_oof_export.to_csv(oof_csv_path, index=False, encoding="utf-8")
    print(f"\n[+] Đã lưu OOF predictions: {oof_csv_path} ({len(df_oof_export)} mẫu)")

    # 4. Phân tích các trường hợp dự đoán sai
    df_misclassified = df_oof_full[df_oof_full["true_intent"] != df_oof_full["predicted_intent"]].copy()
    misclassified_csv_path = output_cv_dir / "misclassified_samples.csv"
    df_mis_export = df_misclassified[["question_id", "question", "true_intent", "predicted_intent", "confidence", "fold"]]
    df_mis_export.to_csv(misclassified_csv_path, index=False, encoding="utf-8")
    print(f"[+] Đã lưu các mẫu dự đoán sai: {misclassified_csv_path} ({len(df_mis_export)} mẫu)")

    # 5. Đánh giá toàn bộ tập OOF
    oof_metrics = compute_metrics(
        y_true=df_oof_full["true_intent"].tolist(),
        y_pred=df_oof_full["predicted_intent"].tolist(),
        labels=INTENT_CLASSES,
    )

    oof_acc = oof_metrics["accuracy"]
    oof_macro_f1 = oof_metrics["macro_f1"]

    acc_list = [fr["accuracy"] for fr in fold_results]
    f1_list = [fr["macro_f1"] for fr in fold_results]
    mean_acc = float(np.mean(acc_list))
    std_acc = float(np.std(acc_list))
    mean_f1 = float(np.mean(f1_list))
    std_f1 = float(np.std(f1_list))

    print(f"\n--- KẾT QUẢ TRUNG BÌNH 3 FOLDS ---")
    print(f"Mean Accuracy:       {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Macro-F1:       {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Overall OOF Acc:     {oof_acc:.4f}")
    print(f"Overall OOF Macro-F1:{oof_macro_f1:.4f}")

    # Vẽ và lưu Confusion Matrix OOF
    cm_plot_path = output_cv_dir / "oof_confusion_matrix.png"
    plot_confusion_matrix(
        cm=np.array(oof_metrics["confusion_matrix"]),
        class_names=INTENT_CLASSES,
        output_path=cm_plot_path,
        title="Out-of-Fold Confusion Matrix (TextCNN)",
    )
    print(f"[+] Đã lưu Confusion Matrix: {cm_plot_path}")

    # Lưu Classification Report text
    report_text_path = output_cv_dir / "oof_classification_report.txt"
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("=== OUT-OF-FOLD CLASSIFICATION REPORT (TextCNN) ===\n\n")
        f.write(oof_metrics["classification_report"])
        f.write(f"\nMean Fold Accuracy:   {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean Fold Macro-F1:   {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Overall OOF Accuracy: {oof_acc:.4f}\n")
        f.write(f"Overall OOF Macro-F1: {oof_macro_f1:.4f}\n")
    print(f"[+] Đã lưu Classification Report: {report_text_path}")

    # Lưu cv_summary.json
    cv_summary = {
        "folds": fold_results,
        "mean_accuracy": round(mean_acc, 4),
        "std_accuracy": round(std_acc, 4),
        "mean_macro_f1": round(mean_f1, 4),
        "std_macro_f1": round(std_f1, 4),
        "overall_oof_accuracy": round(oof_acc, 4),
        "overall_oof_macro_f1": round(oof_macro_f1, 4),
        "per_class_metrics": oof_metrics["per_class"],
        "confusion_matrix": oof_metrics["confusion_matrix"],
        "misclassified_count": len(df_misclassified),
    }

    summary_json_path = output_cv_dir / "cv_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu tóm tắt CV: {summary_json_path}")

    # 6. Kiểm tra model sau khi lưu (Load Test & Verification)
    print("\n" + "=" * 65)
    print("KIỂM TRA CÁC MODEL VÀ PREPROCESSOR ĐÃ LƯU (LOAD TEST)")
    print("=" * 65)

    for fold in range(3):
        model_path = models_dir / f"fold_{fold}.keras"
        prep_path = models_dir / f"fold_{fold}_preprocessor.json"

        # Load lại model và preprocessor
        loaded_model = keras.models.load_model(model_path)
        loaded_prep = IntentPreprocessor.load(prep_path)

        # Lấy lại dữ liệu val của fold
        df_val_fold = df_merged[df_merged["fold"] == fold]
        val_texts = df_val_fold["question"].tolist()
        val_labels = df_val_fold["intent"].tolist()

        # Transform lại bằng loaded preprocessor
        X_val_reload, _ = loaded_prep.transform(val_texts, val_labels)

        # Chạy forward pass
        reload_probs = loaded_model.predict(X_val_reload, verbose=0)

        # Kiểm tra shape
        assert reload_probs.shape == (len(val_texts), 5), f"Shape sai: {reload_probs.shape}"
        # Kiểm tra NaN/Inf
        assert not np.isnan(reload_probs).any(), f"Fold {fold} có NaN sau khi load!"
        assert not np.isinf(reload_probs).any(), f"Fold {fold} có Inf sau khi load!"

        # So sánh với OOF predictions đã lưu
        df_fold_oof = df_oof_full[df_oof_full["fold"] == fold].reset_index(drop=True)
        reloaded_pred_ids = np.argmax(reload_probs, axis=-1)
        reloaded_pred_labels = loaded_prep.decode_labels(reloaded_pred_ids)

        assert reloaded_pred_labels == df_fold_oof["predicted_intent"].tolist(), (
            f"Fold {fold}: Dự đoán sau khi load không khớp với kết quả OOF!"
        )

        print(f"  [+] Fold {fold}: Load thành công, tái tạo chính xác 100% kết quả dự đoán.")

    print("\n" + "=" * 65)
    print("HOÀN THÀNH HUẤN LUYỆN VÀ ĐÁNH GIÁ 3-FOLD CROSS-VALIDATION")
    print("=" * 65)

    return cv_summary


if __name__ == "__main__":
    train_3fold_cross_validation()

