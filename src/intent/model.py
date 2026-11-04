"""
Kiến trúc mô hình TextCNN cho bài toán Intent Classification.
EDGE AI CHALLENGE 2026 – Warm-up

Mô tả kiến trúc TextCNN gọn nhẹ (phù hợp Edge AI / TFLite INT8 / TFLite Micro):
  1. Input layer: chuỗi integer token IDs (batch_size, max_length).
  2. Embedding layer: chuyển đổi token ID thành vector dense (embedding_dim = 32).
  3. Conv1D branches: 3 nhánh tích chập 1D song song với kernel size 2, 3, 4 (mỗi nhánh 16 filters, ReLU).
  4. GlobalMaxPooling1D: trích xuất đặc trưng cực đại theo chiều thời gian từ mỗi nhánh Conv1D.
  5. Concatenate: ghép nối các vector đặc trưng từ 3 nhánh (16 * 3 = 48 đặc trưng).
  6. Dropout: tỷ lệ dropout 0.3 để chống overfitting trên tập dữ liệu nhỏ.
  7. Dense + Softmax: 5 neurons phân loại 5 intent tương ứng.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_textcnn(
    vocab_size: int,
    max_length: int = 24,
    embedding_dim: int = 32,
    num_filters: int = 16,
    kernel_sizes: Optional[List[int]] = None,
    dropout: float = 0.3,
    num_classes: int = 5,
    learning_rate: float = 0.001,
    compile_model: bool = True,
) -> keras.Model:
    """
    Xây dựng mô hình TextCNN nhỏ gọn bằng TensorFlow/Keras Functional API.

    Tham số:
      vocab_size: Kích thước từ vựng (bao gồm cả <pad> và <unk>).
      max_length: Độ dài cố định của chuỗi đầu vào (mặc định 24).
      embedding_dim: Số chiều vector nhúng (mặc định 32).
      num_filters: Số bộ lọc tích chập trên mỗi nhánh Conv1D (mặc định 16).
      kernel_sizes: Danh sách kích thước cửa sổ tích chập (mặc định [2, 3, 4]).
      dropout: Tỷ lệ dropout trước lớp Dense cuối (mặc định 0.3).
      num_classes: Số lượng lớp nhãn đầu ra (mặc định 5).
      learning_rate: Tốc độ học cho optimizer Adam (mặc định 0.001).
      compile_model: Có biên dịch mô hình với optimizer/loss hay không (mặc định True).

    Trả về:
      Đối tượng keras.Model đã được cấu hình và biên dịch.
    """
    if kernel_sizes is None:
        kernel_sizes = [2, 3, 4]

    # 1. Input Layer
    inputs = layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")

    # 2. Embedding Layer
    # Lưu ý: Không dùng mask_zero=True vì các phiên bản TFLite/TFLite Micro cơ bản
    # không hỗ trợ Masking tensor trong các toán tử Conv1D.
    # Token pad (id=0) sẽ được học biểu diễn bình thường.
    embedding = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="embedding",
    )(inputs)

    # 3. Các nhánh Conv1D song song + GlobalMaxPooling1D
    pooled_branches = []
    for k in kernel_sizes:
        conv = layers.Conv1D(
            filters=num_filters,
            kernel_size=k,
            activation="relu",
            padding="valid",
            name=f"conv1d_k{k}",
        )(embedding)
        pool = layers.GlobalMaxPooling1D(name=f"global_max_pool_k{k}")(conv)
        pooled_branches.append(pool)

    # 4. Concatenate các nhánh
    if len(pooled_branches) > 1:
        merged = layers.Concatenate(name="concat_branches")(pooled_branches)
    else:
        merged = pooled_branches[0]

    # 5. Dropout
    dropped = layers.Dropout(rate=dropout, name="dropout")(merged)

    # 6. Dense Output + Softmax
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="probabilities",
    )(dropped)

    model = keras.Model(inputs=inputs, outputs=outputs, name="TextCNN_Intent")

    # 7. Biên dịch mô hình (Compile)
    if compile_model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )

    return model


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Đọc file cấu hình YAML.
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def build_textcnn_from_config(
    config_path: Union[str, Path, Dict[str, Any]],
    vocab_size: int,
    compile_model: bool = True,
) -> keras.Model:
    """
    Khởi tạo mô hình TextCNN từ file cấu hình YAML hoặc dict cấu hình.
    """
    if isinstance(config_path, (str, Path)):
        cfg = load_config(config_path)
    else:
        cfg = dict(config_path)

    return build_textcnn(
        vocab_size=vocab_size,
        max_length=int(cfg.get("max_length", 24)),
        embedding_dim=int(cfg.get("embedding_dim", 32)),
        num_filters=int(cfg.get("num_filters", 16)),
        kernel_sizes=list(cfg.get("kernel_sizes", [2, 3, 4])),
        dropout=float(cfg.get("dropout", 0.3)),
        num_classes=int(cfg.get("num_classes", 5)),
        learning_rate=float(cfg.get("learning_rate", 0.001)),
        compile_model=compile_model,
    )

