"""
Phân tích và khám phá dữ liệu Object Detection (EDA)
EDGE AI CHALLENGE 2026 – Warm-up
"""

import csv
from pathlib import Path
import random
import sys
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Đảm bảo hiển thị Unicode tiếng Việt trên Windows console
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thiết lập đường dẫn gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_utils import load_classes, load_yolo_label, yolo_to_xyxy


def analyze_detection():
    data_raw = PROJECT_ROOT / "data" / "raw"
    train_images_dir = data_raw / "train" / "images"
    train_labels_dir = data_raw / "train" / "labels"
    classes_file = data_raw / "classes.txt"

    output_eda_dir = PROJECT_ROOT / "outputs" / "eda"
    output_fig_dir = PROJECT_ROOT / "outputs" / "figures"
    output_eda_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHÂN TÍCH TẬP DỮ LIỆU OBJECT DETECTION - TRAIN SET")
    print("=" * 60)

    # 1. Đọc danh sách classes
    classes = load_classes(classes_file)
    num_classes = len(classes)
    print(f"\n[1] Danh sách {num_classes} classes từ classes.txt:")
    for idx, cname in enumerate(classes):
        print(f"    ID {idx}: {cname}")

    # Lấy danh sách ảnh train và sắp xếp theo tên
    image_extensions = {".png", ".jpg", ".jpeg"}
    train_image_files = sorted([f for f in train_images_dir.iterdir() if f.suffix.lower() in image_extensions])
    total_images = len(train_image_files)

    # 2. Kiểm tra kích thước ảnh (Width, Height)
    image_sizes = []
    widths = []
    heights = []

    for img_path in train_image_files:
        with Image.open(img_path) as img:
            w, h = img.size
            image_sizes.append((img_path.name, w, h))
            widths.append(w)
            heights.append(h)

    # Lưu image_sizes.csv
    image_sizes_csv = output_eda_dir / "image_sizes.csv"
    with open(image_sizes_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "width", "height"])
        writer.writerows(image_sizes)
    print(f"\n[+] Đã lưu: {image_sizes_csv}")

    size_counter = Counter([(w, h) for _, w, h in image_sizes])
    unique_sizes_count = len(size_counter)
    most_common_size, most_common_count = size_counter.most_common(1)[0]
    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)

    print("\n--- A. THỐNG KÊ KÍCH THƯỚC ẢNH ---")
    print(f"Tổng số ảnh: {total_images}")
    print(f"Số lượng kích thước ảnh khác nhau: {unique_sizes_count}")
    print(f"Kích thước phổ biến nhất: {most_common_size[0]}x{most_common_size[1]} ({most_common_count}/{total_images} ảnh)")
    print(f"Min width: {min_w}, Max width: {max_w}")
    print(f"Min height: {min_h}, Max height: {max_h}")

    # 3. Đọc toàn bộ nhãn YOLO và thống kê
    class_object_counts = {cid: 0 for cid in range(num_classes)}
    class_images_containing = {cid: set() for cid in range(num_classes)}
    objects_per_image = []
    empty_label_images = []
    all_bbox_areas = []

    for img_path in train_image_files:
        stem = img_path.stem
        label_path = train_labels_dir / f"{stem}.txt"
        annotations = load_yolo_label(label_path)
        num_objs = len(annotations)
        objects_per_image.append((img_path.name, num_objs))

        if num_objs == 0:
            empty_label_images.append(img_path.name)
        else:
            for cid, cx, cy, w, h in annotations:
                if 0 <= cid < num_classes:
                    class_object_counts[cid] += 1
                    class_images_containing[cid].add(img_path.name)
                area = w * h
                all_bbox_areas.append(area)

    # Lưu empty_label_images.txt
    empty_label_txt = output_eda_dir / "empty_label_images.txt"
    with open(empty_label_txt, "w", encoding="utf-8") as f:
        for name in empty_label_images:
            f.write(f"{name}\n")
    print(f"[+] Đã lưu: {empty_label_txt}")

    # Lưu objects_per_image.csv
    objects_per_image_csv = output_eda_dir / "objects_per_image.csv"
    with open(objects_per_image_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "num_objects"])
        writer.writerows(objects_per_image)
    print(f"[+] Đã lưu: {objects_per_image_csv}")

    # Lưu detection_class_distribution.csv
    class_dist_csv = output_eda_dir / "detection_class_distribution.csv"
    with open(class_dist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "object_count", "images_containing_class"])
        for cid in range(num_classes):
            writer.writerow([
                cid,
                classes[cid],
                class_object_counts[cid],
                len(class_images_containing[cid])
            ])
    print(f"[+] Đã lưu: {class_dist_csv}")

    # In bảng B. Thống kê object theo class
    print("\n--- B. THỐNG KÊ OBJECT THEO CLASS ---")
    print(f"{'ID':<4} {'Class':<24} {'Objects':<10} {'Images':<8}")
    print("-" * 50)
    for cid in range(num_classes):
        cname = classes[cid]
        objs = class_object_counts[cid]
        imgs = len(class_images_containing[cid])
        print(f"{cid:<4} {cname:<24} {objs:<10} {imgs:<8}")

    # C. Tổng object và trung bình
    counts_list = [n for _, n in objects_per_image]
    total_objects = sum(counts_list)
    avg_objects = total_objects / total_images if total_images > 0 else 0
    min_objects = min(counts_list)
    max_objects = max(counts_list)
    median_objects = float(np.median(counts_list))

    print("\n--- C. TỔNG SỐ OBJECT TRONG TẬP TRAIN ---")
    print(f"Total objects: {total_objects}")
    print(f"Average objects/image: {avg_objects:.2f}")
    print(f"Min objects/image: {min_objects}")
    print(f"Max objects/image: {max_objects}")

    # D. Ảnh không có object
    print("\n--- D. ẢNH KHÔNG CÓ OBJECT ---")
    print(f"Images without objects: {len(empty_label_images)} / {total_images}")

    # E. Phân bố số object/ảnh
    print("\n--- E. PHÂN BỐ SỐ OBJECT TRÊN TỪNG ẢNH ---")
    print(f"mean:   {avg_objects:.2f}")
    print(f"median: {median_objects:.1f}")
    print(f"min:    {min_objects}")
    print(f"max:    {max_objects}")

    # F. Phân bố kích thước bounding box
    bbox_areas = np.array(all_bbox_areas)
    min_area = float(np.min(bbox_areas)) if len(bbox_areas) > 0 else 0.0
    max_area = float(np.max(bbox_areas)) if len(bbox_areas) > 0 else 0.0
    mean_area = float(np.mean(bbox_areas)) if len(bbox_areas) > 0 else 0.0
    median_area = float(np.median(bbox_areas)) if len(bbox_areas) > 0 else 0.0

    small_count = int(np.sum(bbox_areas < 0.02))
    medium_count = int(np.sum((bbox_areas >= 0.02) & (bbox_areas < 0.15)))
    large_count = int(np.sum(bbox_areas >= 0.15))

    print("\n--- F. PHÂN BỐ KÍCH THƯỚC BOUNDING BOX (NORMALIZED AREA = w * h) ---")
    print(f"min bbox area:    {min_area:.6f}")
    print(f"max bbox area:    {max_area:.6f}")
    print(f"mean bbox area:   {mean_area:.6f}")
    print(f"median bbox area: {median_area:.6f}")
    print("\nPhân loại tương đối:")
    print(f"small  (area < 0.02):        {small_count:>4} ({small_count/total_objects*100:.1f}%)")
    print(f"medium (0.02 <= area < 0.15): {medium_count:>4} ({medium_count/total_objects*100:.1f}%)")
    print(f"large  (area >= 0.15):       {large_count:>4} ({large_count/total_objects*100:.1f}%)")

    # 4. Vẽ biểu đồ Detection (Không dùng seaborn, không chỉ định màu thủ công)
    print("\n[+] Đang vẽ biểu đồ Detection...")

    # Biểu đồ 1: Bar chart detection_class_distribution.png
    plt.figure(figsize=(10, 6))
    plt.bar(classes, [class_object_counts[cid] for cid in range(num_classes)])
    plt.xticks(rotation=40, ha="right")
    plt.xlabel("Class Name")
    plt.ylabel("Number of Objects")
    plt.title("Detection Class Distribution (Train Set)")
    plt.tight_layout()
    fig1_path = output_fig_dir / "detection_class_distribution.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu: {fig1_path}")

    # Biểu đồ 2: Histogram objects_per_image_histogram.png
    plt.figure(figsize=(9, 5))
    bins = range(min_objects, max_objects + 2)
    plt.hist(counts_list, bins=bins, align="left", rwidth=0.8)
    plt.xlabel("Number of Objects per Image")
    plt.ylabel("Number of Images")
    plt.title("Objects per Image Distribution (Train Set)")
    plt.tight_layout()
    fig2_path = output_fig_dir / "objects_per_image_histogram.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu: {fig2_path}")

    # Biểu đồ 3: Histogram bbox_area_distribution.png
    plt.figure(figsize=(9, 5))
    plt.hist(all_bbox_areas, bins=35)
    plt.xlabel("Normalized Bounding Box Area (w * h)")
    plt.ylabel("Frequency")
    plt.title("Normalized Bounding Box Area Distribution")
    plt.tight_layout()
    fig3_path = output_fig_dir / "bbox_area_distribution.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"    -> Đã lưu: {fig3_path}")

    # 5. Tạo contact sheet kiểm tra label (12 ảnh ngẫu nhiên với random.seed(42))
    print("\n[+] Đang tạo contact sheet 12 ảnh ngẫu nhiên (random.seed(42))...")
    random.seed(42)
    selected_image_files = random.sample(train_image_files, 12)

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()

    for idx, img_path in enumerate(selected_image_files):
        # Đọc ảnh gốc bằng numpy và cv2 để tránh lỗi đường dẫn
        img_data = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        label_path = train_labels_dir / f"{img_path.stem}.txt"
        annotations = load_yolo_label(label_path)

        annotated = img_rgb.copy()
        for cid, cx, cy, w, h in annotations:
            x_min, y_min, x_max, y_max = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
            x1 = max(0, int(round(x_min)))
            y1 = max(0, int(round(y_min)))
            x2 = min(img_w - 1, int(round(x_max)))
            y2 = min(img_h - 1, int(round(y_max)))

            # Vẽ khung bbox (màu xanh lá/đỏ bằng RGB)
            cname = classes[cid] if 0 <= cid < num_classes else f"class_{cid}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{cid}:{cname}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        ax = axes[idx]
        ax.imshow(annotated)
        ax.set_title(f"{img_path.name} ({len(annotations)} objs)", fontsize=10)
        ax.axis("off")

    plt.suptitle("Contact Sheet: 12 Random Train Samples with YOLO Labels (Seed=42)", fontsize=14, y=0.99)
    plt.tight_layout()
    contact_sheet_path = output_fig_dir / "detection_samples_12.png"
    plt.savefig(contact_sheet_path, dpi=200)
    plt.close()
    print(f"    -> Đã lưu contact sheet: {contact_sheet_path}")

    # 6. Cập nhật outputs/eda/eda_summary.txt với phần [DETECTION]
    eda_summary_path = output_eda_dir / "eda_summary.txt"
    existing_intent = ""
    if eda_summary_path.exists():
        with open(eda_summary_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "[INTENT]" in content:
                existing_intent = "[INTENT]" + content.split("[INTENT]")[1]

    detection_lines = [
        "=== EDGE AI CHALLENGE 2026 - EDA SUMMARY ===",
        "",
        "[DETECTION]",
        "",
        f"Train images: {total_images}",
        f"Total objects: {total_objects}",
        f"Empty images: {len(empty_label_images)} / {total_images}",
        "",
        "Image sizes:",
        f"unique sizes: {unique_sizes_count}",
        f"most common: {most_common_size[0]}x{most_common_size[1]} ({most_common_count}/{total_images} images)",
        f"width range: [{min_w}, {max_w}]",
        f"height range: [{min_h}, {max_h}]",
        "",
        "Objects by class:",
    ]

    for cid in range(num_classes):
        cname = classes[cid]
        objs = class_object_counts[cid]
        imgs = len(class_images_containing[cid])
        pct = (objs / total_objects * 100) if total_objects > 0 else 0
        detection_lines.append(f"  {cid}: {cname:<22} - {objs:>4} objects ({pct:>5.1f}%) in {imgs:>3} images")

    detection_lines.extend([
        "",
        "Objects/image:",
        f"mean:   {avg_objects:.2f}",
        f"median: {median_objects:.1f}",
        f"min:    {min_objects}",
        f"max:    {max_objects}",
        "",
        "BBox:",
        f"small (< 0.02):         {small_count:>4} ({small_count/total_objects*100:.1f}%)",
        f"medium (0.02 <= a < 0.15): {medium_count:>4} ({medium_count/total_objects*100:.1f}%)",
        f"large (>= 0.15):        {large_count:>4} ({large_count/total_objects*100:.1f}%)",
    ])

    if existing_intent:
        summary_text = "\n".join(detection_lines) + "\n\n" + existing_intent
    else:
        summary_text = "\n".join(detection_lines) + "\n"

    with open(eda_summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n[+] Đã cập nhật file báo cáo tổng hợp: {eda_summary_path}")

    return {
        "total_images": total_images,
        "total_objects": total_objects,
        "empty_images_count": len(empty_label_images),
        "unique_sizes_count": unique_sizes_count,
        "most_common_size": most_common_size,
        "most_common_count": most_common_count,
        "min_w": min_w,
        "max_w": max_w,
        "min_h": min_h,
        "max_h": max_h,
        "class_object_counts": class_object_counts,
        "class_images_containing": {cid: len(imgs) for cid, imgs in class_images_containing.items()},
        "classes": classes,
        "avg_objects": avg_objects,
        "median_objects": median_objects,
        "min_objects": min_objects,
        "max_objects": max_objects,
        "small_count": small_count,
        "medium_count": medium_count,
        "large_count": large_count,
    }


if __name__ == "__main__":
    analyze_detection()
