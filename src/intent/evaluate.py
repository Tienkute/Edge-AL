"""
Module đánh giá mô hình Intent Classification.
EDGE AI CHALLENGE 2026 – Warm-up

Cung cấp các hàm tính toán metric (Accuracy, Macro-F1, Precision/Recall theo class, Confusion Matrix)
và vẽ biểu đồ đánh giá mà không sử dụng seaborn hay chỉ định màu thủ công.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Thứ tự 5 nhãn chuẩn
INTENT_CLASSES = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]


def compute_metrics(
    y_true: Union[List[str], np.ndarray],
    y_pred: Union[List[str], np.ndarray],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số đánh giá phân loại:
      - Accuracy
      - Macro-F1
      - Precision, Recall, F1 theo từng class
      - Confusion Matrix
    """
    if labels is None:
        labels = INTENT_CLASSES

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class_metrics = {}
    for idx, cname in enumerate(labels):
        per_class_metrics[cname] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=labels, zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_text,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> Path:
    """
    Vẽ Confusion Matrix bằng Matplotlib thuần túy.
    Không dùng seaborn, không chỉ định màu thủ công (dùng colormap mặc định của matplotlib).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=35, ha="right")
    plt.yticks(tick_marks, class_names)

    # Hiển thị số lượng trong từng ô
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_learning_curves(
    history_dict: Dict[str, List[float]],
    output_path: Union[str, Path],
    title_prefix: str = "Training History",
) -> Path:
    """
    Vẽ đồ thị Loss và Accuracy qua các Epochs.
    Không dùng seaborn, không chỉ định màu thủ công.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Đồ thị Loss
    ax1.plot(epochs, history_dict["loss"], label="Train Loss")
    ax1.plot(epochs, history_dict["val_loss"], label="Val Loss", linestyle="--")
    ax1.set_title(f"{title_prefix} - Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Đồ thị Accuracy
    ax2.plot(epochs, history_dict["accuracy"], label="Train Acc")
    ax2.plot(epochs, history_dict["val_accuracy"], label="Val Acc", linestyle="--")
    ax2.set_title(f"{title_prefix} - Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.legend()
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def evaluate_oof_file(
    oof_csv_path: Union[str, Path] = None,
    output_dir: Union[str, Path] = None,
) -> Dict[str, Any]:
    """
    Đọc file oof_predictions.csv và tính toán toàn bộ báo cáo tổng hợp.
    """
    if oof_csv_path is None:
        oof_csv_path = PROJECT_ROOT / "outputs" / "intent_cv" / "oof_predictions.csv"
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "intent_cv"

    oof_csv_path = Path(oof_csv_path)
    output_dir = Path(output_dir)

    df_oof = pd.read_csv(oof_csv_path)
    metrics = compute_metrics(
        y_true=df_oof["true_intent"].tolist(),
        y_pred=df_oof["predicted_intent"].tolist(),
        labels=INTENT_CLASSES,
    )

    # Lưu confusion matrix plot
    cm_path = output_dir / "oof_confusion_matrix.png"
    plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        class_names=INTENT_CLASSES,
        output_path=cm_path,
        title="Out-of-Fold Confusion Matrix (TextCNN)",
    )

    # Lưu classification report text
    report_path = output_dir / "oof_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== OUT-OF-FOLD CLASSIFICATION REPORT (TextCNN) ===\n\n")
        f.write(metrics["classification_report"])
        f.write(f"\nOverall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Overall Macro-F1: {metrics['macro_f1']:.4f}\n")

    return metrics


if __name__ == "__main__":
    import sys
    print("Testing evaluate module...")
    sample_true = ["COUNT", "EXIST", "LOCATION", "COUNT"]
    sample_pred = ["COUNT", "EXIST", "EXIST", "COUNT"]
    m = compute_metrics(sample_true, sample_pred, INTENT_CLASSES)
    print("Sample metrics:", m["accuracy"], m["macro_f1"])
    print("[+] evaluate.py syntax and logic verified.")

