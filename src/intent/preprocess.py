"""
Module tiền xử lý (preprocessing) dữ liệu Intent Classification.
EDGE AI CHALLENGE 2026 – Warm-up

Quy trình:
  1. Chuẩn hóa chuỗi tiếng Việt không dấu: lowercase, tách/loại bỏ dấu câu, chuẩn hóa khoảng trắng.
  2. Tokenize theo từ (whitespace tokenization).
  3. Xây dựng Vocabulary CHỈ từ tập train của từng fold (PAD=<pad>:0, OOV=<unk>:1).
  4. Chuyển đổi chuỗi từ sang chuỗi số nguyên ID.
  5. Padding/Truncation về độ dài cố định max_length (mặc định 24).
  6. Mã hóa 5 nhãn intent thành số nguyên 0..4.
  7. Hỗ trợ lưu/nạp cấu hình tiền xử lý để tái lập kết quả khi inference.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Các ký hiệu đặc biệt
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<unk>"
PAD_ID = 0
OOV_ID = 1

# Bảng mã hóa 5 nhãn intent cố định theo thứ tự bảng chữ cái
LABEL_TO_ID = {
    "COUNT": 0,
    "EXIST": 1,
    "LIST": 2,
    "LOCATION": 3,
    "OUT_OF_SCOPE": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa câu văn bản tiếng Việt:
      - Chuyển thành chữ thường.
      - Thay thế dấu câu và ký tự đặc biệt bằng khoảng trắng.
      - Gộp các khoảng trắng liên tiếp thành một khoảng trắng duy nhất.
      - Loại bỏ khoảng trắng ở đầu và cuối chuỗi.

    Ví dụ:
      'Co xe tai trong anh khong?' -> 'co xe tai trong anh khong'
      'Xe dap nam o ben trai, o giua hay ben phai?' -> 'xe dap nam o ben trai o giua hay ben phai'
    """
    if text is None:
        return ""
    text = str(text).lower()
    # Thay thế dấu câu bằng khoảng trắng để tránh dính từ (ví dụ "khong?" -> "khong")
    text = re.sub(r"[^\w\s]", " ", text)
    # Chuẩn hóa khoảng trắng dư thừa
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """
    Tách từ đơn giản bằng khoảng trắng sau khi đã chuẩn hóa câu.
    """
    normalized = normalize_text(text)
    return normalized.split() if normalized else []


def pad_sequences(
    sequences: List[List[int]],
    max_length: int = 24,
    padding: str = "post",
    truncating: str = "post",
    pad_value: int = PAD_ID,
) -> np.ndarray:
    """
    Padding hoặc cắt chuỗi số nguyên (token IDs) về độ dài cố định max_length.

    Tham số:
      sequences: Danh sách các chuỗi token IDs.
      max_length: Độ dài cố định của chuỗi (mặc định 24).
      padding: 'post' (thêm pad vào cuối) hoặc 'pre' (thêm vào đầu). Mặc định 'post'.
      truncating: 'post' (cắt ở cuối) hoặc 'pre' (cắt ở đầu). Mặc định 'post'.
      pad_value: Giá trị ID dùng để pad (mặc định 0 cho <pad>).

    Trả về:
      Numpy array có shape (len(sequences), max_length) kiểu np.int32.
    """
    batch_size = len(sequences)
    padded = np.full((batch_size, max_length), pad_value, dtype=np.int32)

    for i, seq in enumerate(sequences):
        if not seq:
            continue
        # Truncate nếu dài hơn max_length
        if len(seq) > max_length:
            if truncating == "post":
                seq_arr = seq[:max_length]
            else:
                seq_arr = seq[-max_length:]
        else:
            seq_arr = seq

        # Gán vào mảng padded
        length = len(seq_arr)
        if padding == "post":
            padded[i, :length] = seq_arr
        else:
            padded[i, max_length - length:] = seq_arr

    return padded


class IntentPreprocessor:
    """
    Lớp quản lý toàn bộ quy trình tiền xử lý câu hỏi và nhãn cho Intent Classification.
    Đảm bảo tính độc lập: chỉ fit vocabulary trên tập train để chống rò rỉ dữ liệu (data leakage).
    """

    def __init__(self, max_length: int = 24):
        self.max_length = max_length
        self.pad_token = PAD_TOKEN
        self.oov_token = OOV_TOKEN
        self.pad_id = PAD_ID
        self.oov_id = OOV_ID
        self.label_to_id: Dict[str, int] = dict(LABEL_TO_ID)
        self.id_to_label: Dict[int, str] = dict(ID_TO_LABEL)

        # Khởi tạo vocab với PAD và OOV
        self.vocab: Dict[str, int] = {
            self.pad_token: self.pad_id,
            self.oov_token: self.oov_id,
        }
        self.id_to_vocab: Dict[int, str] = {
            self.pad_id: self.pad_token,
            self.oov_id: self.oov_token,
        }
        self.is_fitted: bool = False

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_classes(self) -> int:
        return len(self.label_to_id)

    def fit(self, texts: List[str]) -> "IntentPreprocessor":
        """
        Xây dựng vocabulary từ danh sách các câu hỏi của tập huấn luyện (training set).
        CẢNH BÁO: Không bao giờ fit trên tập validation hoặc test!
        """
        # Reset vocab về trạng thái ban đầu
        self.vocab = {
            self.pad_token: self.pad_id,
            self.oov_token: self.oov_id,
        }

        # Thu thập các từ duy nhất, sắp xếp để đảm bảo tính nhất quán (deterministic)
        unique_words = set()
        for text in texts:
            words = tokenize(text)
            unique_words.update(words)

        sorted_words = sorted(unique_words)
        for idx, word in enumerate(sorted_words, start=2):
            self.vocab[word] = idx

        self.id_to_vocab = {v: k for k, v in self.vocab.items()}
        self.is_fitted = True
        return self

    def text_to_sequence(self, text: str) -> List[int]:
        """
        Chuyển một câu văn bản thành chuỗi integer IDs dựa trên vocabulary hiện tại.
        Nếu từ không có trong vocab, thay thế bằng OOV_ID (<unk>).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor chưa được fit! Hãy gọi fit() trên tập train trước.")
        words = tokenize(text)
        return [self.vocab.get(w, self.oov_id) for w in words]

    def transform_texts(self, texts: List[str]) -> np.ndarray:
        """
        Chuyển danh sách văn bản thành mảng 2D numpy (N, max_length) chứa token IDs.
        """
        sequences = [self.text_to_sequence(t) for t in texts]
        return pad_sequences(
            sequences,
            max_length=self.max_length,
            padding="post",
            truncating="post",
            pad_value=self.pad_id,
        )

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Mã hóa danh sách nhãn text (COUNT, EXIST, ...) thành mảng số nguyên 0..4.
        """
        encoded = []
        for lbl in labels:
            if lbl not in self.label_to_id:
                raise ValueError(f"Nhãn không hợp lệ: '{lbl}'. Phải thuộc {list(self.label_to_id.keys())}")
            encoded.append(self.label_to_id[lbl])
        return np.array(encoded, dtype=np.int32)

    def decode_labels(self, label_ids: Union[List[int], np.ndarray]) -> List[str]:
        """
        Chuyển đổi các ID dự đoán (0..4) ngược lại thành tên nhãn text.
        """
        return [self.id_to_label[int(lid)] for lid in label_ids]

    def transform(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Biến đổi đồng thời texts (và labels nếu có).
        """
        X = self.transform_texts(texts)
        if labels is not None:
            y = self.encode_labels(labels)
            return X, y
        return X

    def fit_transform(
        self, texts: List[str], labels: Optional[List[str]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Fit trên texts và biến đổi kết quả.
        """
        self.fit(texts)
        return self.transform(texts, labels)

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Lưu toàn bộ trạng thái preprocessor vào file JSON.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_length": self.max_length,
            "pad_token": self.pad_token,
            "oov_token": self.oov_token,
            "pad_id": self.pad_id,
            "oov_id": self.oov_id,
            "vocab": self.vocab,
            "label_to_id": self.label_to_id,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IntentPreprocessor":
        """
        Nạp preprocessor từ file JSON đã lưu.
        """
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(max_length=data["max_length"])
        instance.pad_token = data["pad_token"]
        instance.oov_token = data["oov_token"]
        instance.pad_id = data["pad_id"]
        instance.oov_id = data["oov_id"]
        instance.vocab = {k: int(v) for k, v in data["vocab"].items()}
        instance.id_to_vocab = {int(v): k for k, v in instance.vocab.items()}
        instance.label_to_id = {k: int(v) for k, v in data["label_to_id"].items()}
        instance.id_to_label = {int(v): k for k, v in instance.label_to_id.items()}
        instance.is_fitted = True
        return instance

