"""
Tiện ích xử lý dữ liệu và nhãn định dạng YOLO cho bài toán Object Detection.
"""

from pathlib import Path
from typing import List, Tuple, Union


def load_yolo_label(label_path: Union[str, Path]) -> List[Tuple[int, float, float, float, float]]:
    """
    Đọc file nhãn định dạng YOLO (.txt).

    Mỗi dòng hợp lệ có dạng:
        class_id cx cy w h
    Trong đó:
        class_id: số nguyên định danh lớp đối tượng (0-9)
        cx, cy: tọa độ tâm bbox (chuẩn hóa 0..1)
        w, h: chiều rộng và chiều cao bbox (chuẩn hóa 0..1)

    Trả về:
        Danh sách các tuple: [(class_id, cx, cy, w, h), ...]
        Nếu file rỗng hoặc không tồn tại, trả về danh sách rỗng [].
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    annotations = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            parts = line_str.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                annotations.append((class_id, cx, cy, w, h))

    return annotations


def yolo_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_width: Union[int, float],
    img_height: Union[int, float],
) -> Tuple[float, float, float, float]:
    """
    Chuyển đổi bounding box từ định dạng YOLO (cx, cy, w, h) chuẩn hóa 0..1
    sang tọa độ pixel (x_min, y_min, x_max, y_max) của ảnh gốc.

    Tham số:
        cx: tọa độ tâm theo trục X (chuẩn hóa 0..1)
        cy: tọa độ tâm theo trục Y (chuẩn hóa 0..1)
        w: chiều rộng bbox (chuẩn hóa 0..1)
        h: chiều cao bbox (chuẩn hóa 0..1)
        img_width: chiều rộng thực tế của ảnh (pixel)
        img_height: chiều cao thực tế của ảnh (pixel)

    Trả về:
        Tuple (x_min, y_min, x_max, y_max) dưới dạng float theo pixel,
        chưa làm tròn tọa độ quá sớm để giữ độ chính xác.
    """
    x_min = (cx - w / 2.0) * img_width
    y_min = (cy - h / 2.0) * img_height
    x_max = (cx + w / 2.0) * img_width
    y_max = (cy + h / 2.0) * img_height

    return x_min, y_min, x_max, y_max


def load_classes(classes_path: Union[str, Path]) -> List[str]:
    """
    Đọc danh sách tên các lớp từ file classes.txt.
    """
    classes_path = Path(classes_path)
    if not classes_path.exists():
        return []
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

