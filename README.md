# EDGE AI CHALLENGE 2026 – Warm-up

Dự án chuẩn bị và thực thi các bài toán trong cuộc thi **EDGE AI CHALLENGE 2026 – Warm-up**.

## Giới thiệu bài toán

Dự án bao gồm 2 bài toán chính:
1. **Object Detection**: Phát hiện các đối tượng giao thông (10 lớp) từ dữ liệu ảnh với định dạng nhãn bounding box chuẩn YOLO (`class_id cx cy w h`).
2. **Intent Classification bằng TextCNN**: Phân loại ý định (intent) của các câu hỏi liên quan đến tình huống giao thông (các nhãn intent như EXIST, COUNT, LOCATION, LIST,...).

## Cấu trúc thư mục

```text
Edge_AI/
│
├── configs/
├── data/
│   ├── processed/
│   └── raw/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   └── images/
│       ├── classes.txt
│       ├── intent_train.csv
│       ├── intent_test.csv
│       └── sample_submission.csv
│
├── models/
├── notebooks/
├── outputs/
├── src/
│   ├── detection/
│   ├── intent/
│   ├── submission/
│   └── utils/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Kiểm tra Dataset

Để kiểm tra tính toàn vẹn và hợp lệ của toàn bộ tập dữ liệu (ảnh train/test, nhãn YOLO, các file CSV intent và sample submission), chạy lệnh sau từ thư mục gốc của project:

```bash
python src/utils/check_dataset.py
```
