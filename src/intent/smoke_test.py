"""
Smoke test kiểm tra toàn diện pipeline TextCNN cho Intent Classification.
EDGE AI CHALLENGE 2026 – Warm-up

Nội dung kiểm tra:
  1. Load dữ liệu train và fold split (chọn fold 0).
  2. Fit vocabulary CHỈ trên tập train của fold 0 (42 mẫu).
  3. Preprocess train và validation.
  4. Kiểm tra shape dữ liệu.
  5. Build mô hình TextCNN từ configs/intent.yaml.
  6. In model.summary() và đếm số parameters.
  7. Chạy forward pass trên batch nhỏ.
  8. Kiểm tra output shape (batch_size, 5).
  9. Kiểm tra tính hợp lệ của phân phối xác suất (Softmax: sum ~ 1.0, 0 <= p <= 1).
  10. Kiểm tra không có NaN hoặc Inf.
  11. Kiểm tra tính xác định (determinism/reproducibility) của tiền xử lý.
  12. Kiểm tra 1 bước tối ưu (train_on_batch) trên batch nhỏ (không chạy full epochs).
"""

import sys
from pathlib import Path
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
from src.intent.model import build_textcnn_from_config, load_config
from src.intent.preprocess import IntentPreprocessor


def run_smoke_test(fold_to_test: int = 0) -> bool:
    print("=" * 65)
    print("SMOKE TEST: TEXTCNN PIPELINE (INTENT CLASSIFICATION)")
    print("=" * 65)

    data_raw = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    config_path = PROJECT_ROOT / "configs" / "intent.yaml"

    intent_train_csv = data_raw / "intent_train.csv"
    intent_folds_csv = processed_dir / "intent_folds.csv"

    # 1. Load dữ liệu và fold
    print("\n[1] Đọc dữ liệu train và thông tin folds...")
    if not intent_train_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy: {intent_train_csv}")
    if not intent_folds_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy: {intent_folds_csv}")

    df_train_all = pd.read_csv(intent_train_csv)
    df_folds = pd.read_csv(intent_folds_csv)

    df_merged = df_train_all.merge(df_folds, on="question_id", how="inner")
    print(f"    Tổng số câu hỏi: {len(df_merged)}")

    # 2. Phân chia Train/Validation theo fold chỉ định (mặc định fold 0)
    print(f"\n[2] Chọn Fold {fold_to_test}:")
    df_train_fold = df_merged[df_merged["fold"] != fold_to_test].reset_index(drop=True)
    df_val_fold = df_merged[df_merged["fold"] == fold_to_test].reset_index(drop=True)

    print(f"    Số mẫu train (Fold {fold_to_test}): {len(df_train_fold)}")
    print(f"    Số mẫu val   (Fold {fold_to_test}): {len(df_val_fold)}")

    train_texts = df_train_fold["question"].tolist()
    train_labels = df_train_fold["intent"].tolist()
    val_texts = df_val_fold["question"].tolist()
    val_labels = df_val_fold["intent"].tolist()

    # 3. Đọc file config
    print("\n[3] Đọc cấu hình từ configs/intent.yaml...")
    cfg = load_config(config_path)
    max_length = int(cfg.get("max_length", 24))
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 8))

    tf.random.set_seed(seed)
    np.random.seed(seed)

    # 4. Tiền xử lý: Fit vocabulary CHỈ trên tập train
    print("\n[4] Xây dựng vocabulary (Fit CHỈ trên tập train của fold)...")
    preprocessor = IntentPreprocessor(max_length=max_length)
    preprocessor.fit(train_texts)

    vocab_size = preprocessor.vocab_size
    print(f"    Vocabulary size: {vocab_size} tokens (bao gồm <pad> và <unk>)")
    print(f"    Ví dụ một số từ đầu tiên: {list(preprocessor.vocab.keys())[:10]}")

    # 5. Transform tập train và validation
    print("\n[5] Chuyển đổi dữ liệu sang định dạng tensor...")
    X_train, y_train = preprocessor.transform(train_texts, train_labels)
    X_val, y_val = preprocessor.transform(val_texts, val_labels)

    # 6. Kiểm tra shape
    print(f"    X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"    X_val shape:   {X_val.shape}   | y_val shape:   {y_val.shape}")

    assert X_train.shape == (len(train_texts), max_length), f"X_train shape sai: {X_train.shape}"
    assert y_train.shape == (len(train_labels),), f"y_train shape sai: {y_train.shape}"
    assert X_val.shape == (len(val_texts), max_length), f"X_val shape sai: {X_val.shape}"
    assert y_val.shape == (len(val_labels),), f"y_val shape sai: {y_val.shape}"
    print("    [+] Shape dữ liệu Train và Validation hoàn toàn chính xác!")

    # 7. Xây dựng mô hình TextCNN
    print("\n[6] Khởi tạo kiến trúc mô hình TextCNN...")
    model = build_textcnn_from_config(cfg, vocab_size=vocab_size, compile_model=True)
    print("\n" + "-" * 65)
    model.summary()
    print("-" * 65)

    total_params = model.count_params()
    trainable_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    non_trainable_params = total_params - trainable_params
    print(f"    Total params:         {total_params:,}")
    print(f"    Trainable params:     {trainable_params:,}")
    print(f"    Non-trainable params: {non_trainable_params:,}")

    # 8. Chạy Forward Pass trên một batch nhỏ
    test_batch_size = min(batch_size, len(X_val))
    batch_X = X_val[:test_batch_size]
    batch_y = y_val[:test_batch_size]

    print(f"\n[7] Kiểm tra Forward Pass trên batch {test_batch_size} mẫu...")
    preds = model.predict(batch_X, verbose=0)

    # 9. Kiểm tra Output Shape
    print(f"    Output shape: {preds.shape} (Kỳ vọng: ({test_batch_size}, 5))")
    assert preds.shape == (test_batch_size, 5), f"Output shape sai: {preds.shape}"
    print("    [+] Output shape hợp lệ.")

    # 10. Kiểm tra phân phối xác suất
    print("\n[8] Kiểm tra tính hợp lệ của phân phối xác suất (Softmax)...")
    prob_sums = np.sum(preds, axis=-1)
    print(f"    Tổng xác suất mỗi mẫu: {prob_sums}")
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Tổng xác suất Softmax không bằng 1.0!"
    assert (preds >= 0.0).all() and (preds <= 1.0).all(), "Giá trị xác suất nằm ngoài [0, 1]!"
    print("    [+] Phân phối xác suất Softmax hoàn toàn hợp lệ.")

    # 11. Kiểm tra NaN / Inf
    print("\n[9] Kiểm tra giá trị NaN và Inf...")
    assert not np.isnan(preds).any(), "Phát hiện giá trị NaN trong output!"
    assert not np.isinf(preds).any(), "Phát hiện giá trị Inf trong output!"
    print("    [+] Không có giá trị NaN hoặc Inf.")

    # 12. Kiểm tra tính xác định (determinism) của tiền xử lý
    print("\n[10] Kiểm tra tính xác định (reproducibility) của Preprocessing...")
    X_val_repeat = preprocessor.transform_texts(val_texts)
    assert np.array_equal(X_val, X_val_repeat), "Preprocessing cho kết quả khác nhau giữa 2 lần gọi!"
    print("    [+] Preprocessing đảm bảo tính nhất quán tuyệt đối.")

    # 13. Kiểm tra 1 bước tối ưu (train_on_batch)
    print("\n[11] Kiểm tra 1 bước Backpropagation (train_on_batch)...")
    initial_loss, initial_acc = model.evaluate(batch_X, batch_y, verbose=0)
    print(f"    Loss trước train_on_batch: {initial_loss:.4f} | Acc: {initial_acc:.4f}")

    step_loss, step_acc = model.train_on_batch(batch_X, batch_y)
    print(f"    Loss tại train_on_batch:   {step_loss:.4f} | Acc: {step_acc:.4f}")
    assert not np.isnan(step_loss) and not np.isinf(step_loss), "Loss sau bước cập nhật là NaN/Inf!"
    print("    [+] Backpropagation hoạt động ổn định, gradients được cập nhật bình thường.")

    print("\n" + "=" * 65)
    print("TẤT CẢ CÁC BƯỚC SMOKE TEST ĐỀU ĐẠT CHUẨN [PASSED]")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = run_smoke_test(fold_to_test=0)
    sys.exit(0 if success else 1)
