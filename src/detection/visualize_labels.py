"""
Trực quan hóa bounding box nhãn YOLO lên ảnh huấn luyện.
Lưu ảnh trực quan hóa vào outputs/figures/.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm project root vào sys.path để import yolo_utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_utils import load_classes, load_yolo_label, yolo_to_xyxy


# Bảng màu trực quan cho 10 lớp đối tượng (BGR cho OpenCV)
CLASS_COLORS = [
    (255, 56, 56),    # 0: person - xanh lơ/đỏ nhạt
    (255, 157, 0),   # 1: bicycle - cam
    (255, 112, 31),  # 2: motorcycle - cam đậm
    (72, 249, 10),   # 3: car - xanh lá
    (10, 249, 72),   # 4: bus - xanh lá ngọc
    (20, 204, 255),  # 5: truck - vàng
    (255, 0, 199),   # 6: traffic_light - hồng tím
    (0, 0, 255),     # 7: stop_sign - đỏ
    (0, 165, 255),   # 8: crossing_warning_sign - cam vàng
    (255, 255, 0),   # 9: zebra_crossing - xanh cyan
]


def visualize_image_labels(
    image_name: str = None,
    output_dir: Path = None,
) -> Path:
    """
    Trực quan hóa bounding box của một ảnh train và lưu vào thư mục output_dir.

    Tham số:
        image_name: Tên file ảnh (ví dụ: '0147.png' hoặc '0147').
                    Nếu None, tự động chọn ảnh train đầu tiên có nhãn.
        output_dir: Thư mục lưu kết quả. Mặc định là outputs/figures/.

    Trả về:
        Đường dẫn file ảnh kết quả đã lưu.
    """
    data_raw = PROJECT_ROOT / "data" / "raw"
    train_images_dir = data_raw / "train" / "images"
    train_labels_dir = data_raw / "train" / "labels"
    classes_file = data_raw / "classes.txt"

    # 1. Đọc danh sách classes
    classes = load_classes(classes_file)
    if not classes:
        raise FileNotFoundError(f"Không tìm thấy classes tại {classes_file}")

    # 2. Tạo thư mục outputs/figures nếu chưa tồn tại
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Chọn file ảnh
    if image_name:
        image_stem = Path(image_name).stem
        image_path = train_images_dir / f"{image_stem}.png"
        if not image_path.exists():
            # Thử tìm với tên gốc
            image_path = train_images_dir / image_name
    else:
        # Tự động chọn một ảnh có nhãn chứa nhiều đối tượng (ví dụ: 0147.png nếu có, hoặc ảnh đầu tiên có nhãn)
        preferred = train_images_dir / "0147.png"
        if preferred.exists():
            image_path = preferred
            image_stem = "0147"
        else:
            candidates = sorted(list(train_images_dir.glob("*.png")))
            if not candidates:
                raise FileNotFoundError(f"Không có ảnh trong {train_images_dir}")
            image_path = candidates[0]
            image_stem = image_path.stem

    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    label_path = train_labels_dir / f"{image_path.stem}.txt"

    # 4. Đọc ảnh (dùng np.fromfile và cv2.imdecode để tránh lỗi đường dẫn tiếng Việt nếu có)
    img_data = np.fromfile(str(image_path), dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")

    # Bản sao để vẽ, TUYỆT ĐỐI không sửa ảnh gốc
    annotated_img = img.copy()
    img_h, img_w = annotated_img.shape[:2]

    # 5. Đọc nhãn YOLO
    annotations = load_yolo_label(label_path)
    print(f"Ảnh: {image_path.name} ({img_w}x{img_h}px) - Số object: {len(annotations)}")

    # 6. Vẽ bounding box và class name
    for class_id, cx, cy, w, h in annotations:
        x_min, y_min, x_max, y_max = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)

        # Chuyển đổi sang tọa độ nguyên sau khi tính toán đầy đủ
        x1 = max(0, int(round(x_min)))
        y1 = max(0, int(round(y_min)))
        x2 = min(img_w - 1, int(round(x_max)))
        y2 = min(img_h - 1, int(round(y_max)))

        # Tên class và màu sắc
        class_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

        # Vẽ hình chữ nhật bounding box
        thickness = max(2, int(round(min(img_w, img_h) / 400)))
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

        # Chuẩn bị nhãn text
        label_text = f"{class_id}: {class_name}"
        font_scale = max(0.4, min(img_w, img_h) / 1200)
        font_thickness = max(1, int(thickness / 2))
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Nền cho text để dễ nhìn
        text_y1 = max(0, y1 - th - baseline - 4)
        text_y2 = y1
        text_x2 = min(img_w, x1 + tw + 6)
        cv2.rectangle(annotated_img, (x1, text_y1), (text_x2, text_y2), color, -1)

        # Vẽ chữ (màu trắng hoặc đen tùy nền)
        text_color = (0, 0, 0) if (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299) > 150 else (255, 255, 255)
        cv2.putText(
            annotated_img,
            label_text,
            (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA,
        )

    # 7. Lưu ảnh kết quả vào outputs/figures/
    output_filename = f"{image_path.stem}_annotated.png"
    output_filepath = output_dir / output_filename
    is_success, buffer = cv2.imencode(".png", annotated_img)
    if is_success:
        with open(output_filepath, "wb") as f:
            f.write(buffer)
        print(f"[+] Đã lưu ảnh kết quả vào: {output_filepath}")
    else:
        raise IOError(f"Không thể ghi ảnh ra: {output_filepath}")

    return output_filepath


def main():
    parser = argparse.ArgumentParser(description="Trực quan hóa nhãn YOLO lên ảnh train.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Tên file ảnh train (ví dụ: '0147.png'). Mặc định tự động chọn ảnh.",
    )
    args = parser.parse_args()
    visualize_image_labels(image_name=args.image)


if __name__ == "__main__":
    main()
