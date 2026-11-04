"""
Unit tests cho module tiền xử lý Intent Classification (src/intent/preprocess.py).
EDGE AI CHALLENGE 2026 – Warm-up
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Thêm project root vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.intent.preprocess import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    OOV_ID,
    OOV_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    IntentPreprocessor,
    normalize_text,
    pad_sequences,
    tokenize,
)


class TestIntentPreprocess(unittest.TestCase):
    def setUp(self):
        self.sample_train = [
            "Co xe tai trong anh khong?",
            "Hay dem so bien STOP xuat hien trong canh.",
            "Xe dap nam o ben trai, o giua hay ben phai khung hinh?",
        ]
        self.sample_val = [
            "Vach qua duong co xuat hien trong canh khong?",
            "May bay co bay qua bau troi khong?",  # chứa từ lạ: 'may', 'bay', 'bau', 'troi'
        ]

    def test_normalize_text(self):
        """Kiểm tra chuẩn hóa câu: chữ thường, xóa dấu câu, gộp khoảng trắng."""
        raw = "  Bien canh bao nguoi di bo... qua duong, nam o ben trai???  "
        expected = "bien canh bao nguoi di bo qua duong nam o ben trai"
        self.assertEqual(normalize_text(raw), expected)

        # Kiểm tra None và chuỗi rỗng
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

    def test_tokenize(self):
        """Kiểm tra tách từ sau khi chuẩn hóa."""
        text = "Xe may, xe dap va xe tai!"
        tokens = tokenize(text)
        expected = ["xe", "may", "xe", "dap", "va", "xe", "tai"]
        self.assertEqual(tokens, expected)

    def test_padding(self):
        """Kiểm tra padding chuỗi ngắn về max_length."""
        seqs = [[2, 3, 4], [5]]
        padded = pad_sequences(seqs, max_length=6, pad_value=PAD_ID, padding="post")
        self.assertEqual(padded.shape, (2, 6))
        np.testing.assert_array_equal(padded[0], [2, 3, 4, 0, 0, 0])
        np.testing.assert_array_equal(padded[1], [5, 0, 0, 0, 0, 0])

    def test_truncation(self):
        """Kiểm tra cắt ngắn chuỗi vượt quá max_length."""
        seqs = [[1, 2, 3, 4, 5, 6, 7, 8]]
        truncated = pad_sequences(seqs, max_length=5, truncating="post")
        self.assertEqual(truncated.shape, (1, 5))
        np.testing.assert_array_equal(truncated[0], [1, 2, 3, 4, 5])

    def test_oov_token(self):
        """Kiểm tra các từ không có trong vocabulary được gán nhãn OOV (<unk>)."""
        prep = IntentPreprocessor(max_length=10)
        prep.fit(["xe tai chay tren duong"])

        # Từ 'may' và 'bay' chưa từng xuất hiện trong train
        seq = prep.text_to_sequence("may bay chay tren duong")
        # 'may' -> OOV, 'bay' -> OOV, 'chay' -> train ID, 'tren' -> train ID, 'duong' -> train ID
        self.assertEqual(seq[0], OOV_ID)
        self.assertEqual(seq[1], OOV_ID)
        self.assertNotEqual(seq[2], OOV_ID)
        self.assertNotEqual(seq[3], OOV_ID)
        self.assertNotEqual(seq[4], OOV_ID)

    def test_label_mapping(self):
        """Kiểm tra mã hóa 5 nhãn intent và giải mã 2 chiều."""
        prep = IntentPreprocessor()
        intents = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
        encoded = prep.encode_labels(intents)

        self.assertEqual(len(encoded), 5)
        self.assertTrue((encoded >= 0).all() and (encoded < 5).all())

        decoded = prep.decode_labels(encoded)
        self.assertEqual(decoded, intents)

        # Kiểm tra nhãn không hợp lệ phải raise ValueError
        with self.assertRaises(ValueError):
            prep.encode_labels(["INVALID_LABEL"])

    def test_vocab_no_validation_leakage(self):
        """Kiểm tra vocabulary KHÔNG chứa từ chỉ xuất hiện trong validation nếu từ đó không có trong train."""
        prep = IntentPreprocessor(max_length=24)
        prep.fit(self.sample_train)

        # 'bau' và 'troi' chỉ có trong sample_val, không có trong sample_train
        self.assertNotIn("bau", prep.vocab)
        self.assertNotIn("troi", prep.vocab)

        # Khi transform sample_val, từ 'bau' và 'troi' phải được ánh xạ sang OOV_ID
        seq = prep.text_to_sequence("may bay qua bau troi")
        self.assertIn(OOV_ID, seq)

    def test_save_and_load_consistency(self):
        """Kiểm tra tính nhất quán tuyệt đối giữa preprocessor trước và sau khi lưu/nạp."""
        prep_original = IntentPreprocessor(max_length=16)
        prep_original.fit(self.sample_train)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "preprocessor.json"
            prep_original.save(save_path)

            self.assertTrue(save_path.exists())

            prep_loaded = IntentPreprocessor.load(save_path)

            self.assertEqual(prep_original.max_length, prep_loaded.max_length)
            self.assertEqual(prep_original.vocab, prep_loaded.vocab)
            self.assertEqual(prep_original.label_to_id, prep_loaded.label_to_id)

            # Kiểm tra kết quả transform hoàn toàn trùng khớp
            test_text = ["co bien stop o ben phai"]
            out_orig = prep_original.transform_texts(test_text)
            out_loaded = prep_loaded.transform_texts(test_text)
            np.testing.assert_array_equal(out_orig, out_loaded)

    def test_original_dataset_csv_unmodified(self):
        """Kiểm tra đảm bảo các file CSV gốc không bị sửa đổi."""
        train_csv = PROJECT_ROOT / "data" / "raw" / "intent_train.csv"
        df = pd.read_csv(train_csv)
        self.assertEqual(len(df), 63)
        self.assertListEqual(list(df.columns), ["question_id", "question", "intent"])


if __name__ == "__main__":
    unittest.main()

