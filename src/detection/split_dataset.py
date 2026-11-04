"""
Chia tập dữ liệu Object Detection thành train / validation.
EDGE AI CHALLENGE 2026 – Warm-up

Phương pháp:
  - Tạo ma trận multi-label cho 140 ảnh (mỗi ảnh × 10 class: 0/1).
  - Áp dụng Iterative Stratification (thuật toán tham lam) để chia sao cho
    tỷ lệ từng class được phân bổ đồng đều giữa train và val.
  - Nhận diện nhóm ảnh "cùng cảnh" bằng pHash Hamming distance.
  - Ưu tiên giữ ảnh cùng nhóm trong cùng một tập để tránh leakage.
  - random.seed(42) để tái hiện kết quả.

Lưu:
  data/processed/detection_train.txt
  data/processed/detection_val.txt
  outputs/eda/detection_split_report.csv
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Đảm bảo Unicode trên Windows terminal
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.yolo_utils import load_classes, load_yolo_label

# ============================================================
# Hằng số cấu hình
# ============================================================
RANDOM_SEED = 42
VAL_RATIO = 0.2          # ~28/140 = 20% dùng làm validation
PHASH_SIZE = 8           # Kích thước hash DCT
PHASH_FREQ = 4           # highfreq_factor
NEAR_DUP_HAMMING = 10    # Ngưỡng Hamming để coi là "cùng cảnh"


def compute_phash(img_path: Path, hash_size: int = PHASH_SIZE, highfreq: int = PHASH_FREQ) -> np.ndarray:
    """Tính perceptual hash (pHash) dựa trên DCT."""
    from scipy.fft import dct as scipy_dct
    img_size = hash_size * highfreq
    with Image.open(img_path) as img:
        small = np.array(img.convert("L").resize((img_size, img_size), Image.LANCZOS), dtype=float)
    # Áp dụng DCT 2D
    dct_coef = scipy_dct(scipy_dct(small, axis=0), axis=1)
    top_left = dct_coef[:hash_size, :hash_size]
    median_val = np.median(top_left)
    return (top_left > median_val).flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Khoảng cách Hamming giữa hai pHash."""
    return int(np.sum(h1 != h2))


def build_near_dup_groups(image_files: list, threshold: int = NEAR_DUP_HAMMING) -> dict:
    """
    Tìm các nhóm ảnh "cùng cảnh" dựa trên pHash.
    Trả về dict: {image_name: group_id}.
    Ảnh không thuộc nhóm nào có group_id = None.
    """
    print("  Tính pHash để phát hiện ảnh gần trùng...")
    hashes = {}
    for f in image_files:
        try:
            hashes[f.name] = compute_phash(f)
        except Exception as e:
            print(f"    [!] Không thể tính pHash cho {f.name}: {e}")

    names = list(hashes.keys())
    # Union-Find
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    pairs_found = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dist = hamming_distance(hashes[names[i]], hashes[names[j]])
            if dist <= threshold:
                union(names[i], names[j])
                pairs_found.append((names[i], names[j], dist))

    # Tạo nhóm
    groups = defaultdict(list)
    for name in names:
        groups[find(name)].append(name)

    # Chỉ lấy nhóm có > 1 ảnh
    near_dup_groups = {r: sorted(g) for r, g in groups.items() if len(g) > 1}
    print(f"  -> Phát hiện {len(pairs_found)} cặp ảnh cùng cảnh (Hamming <= {threshold}).")
    print(f"  -> Phát hiện {len(near_dup_groups)} nhóm ảnh gần trùng:")
    group_map = {}  # image_name -> group_id
    for gid, (_, members) in enumerate(near_dup_groups.items()):
        print(f"       Nhóm {gid+1}: {members}")
        for m in members:
            group_map[m] = gid

    return group_map, pairs_found, near_dup_groups


def iterative_stratification_split(
    image_names: list,
    label_matrix: np.ndarray,
    group_map: dict,
    near_dup_groups: dict,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> tuple:
    """
    Chia train/val cho bài toán multi-label detection.

    Phương pháp: Greedy Iterative Stratification (Sechidis et al., 2011).
    
    Thuật toán:
      1. Nhóm hóa ảnh theo near-dup group để đảm bảo cùng nhóm → cùng tập.
      2. Với mỗi đơn vị (unit = ảnh đơn hoặc nhóm near-dup):
         a. Tìm class c* ít xuất hiện nhất còn cần val (remaining_desired_val[c] cao nhất tương đối).
         b. Nếu unit chứa c*: gán vào val với xác suất remaining_desired_val[c*]/remaining_total[c*].
         c. Ngược lại: gán vào tập có tỷ lệ c* thấp hơn.
      3. Nếu val đã đủ số lượng mục tiêu, những unit còn lại vào train.
      
    Ưu tiên: Không để ảnh cùng near-dup group bị chia split (tránh leakage).
    """
    rng = random.Random(seed)
    n = len(image_names)
    n_classes = label_matrix.shape[1]
    target_val_total = round(n * val_ratio)  # Mục tiêu: 28 ảnh val

    # Tính số lượng ảnh chứa mỗi class
    class_totals = label_matrix.sum(axis=0).astype(float)
    # Mục tiêu số ảnh val chứa mỗi class (theo tỷ lệ)
    desired_val = np.round(class_totals * val_ratio).astype(float)

    # --- Nhóm hóa ảnh ---
    # Ảnh cùng near-dup group → một "unit" duy nhất
    group_to_images = defaultdict(list)
    for name in image_names:
        g = group_map.get(name, name)  # Nếu không có group → dùng tên ảnh làm key đơn lẻ
        group_to_images[g].append(name)

    units = [(g, sorted(members)) for g, members in group_to_images.items()]

    # Shuffle units bằng seed để có thứ tự ngẫu nhiên nhất quán
    rng.shuffle(units)

    # Tính nhãn tổng hợp cho từng unit
    unit_labels = []
    for g, members in units:
        combined = label_matrix[[image_names.index(m) for m in members]].max(axis=0)
        unit_labels.append(combined)
    unit_labels = np.array(unit_labels)

    # --- Thuật toán Iterative Stratification ---
    remaining_desired_val = desired_val.copy()
    # Số ảnh chứa mỗi class chưa được gán
    remaining_total = class_totals.copy()

    val_units = []
    train_units = []
    val_count = 0

    n_units = len(units)

    for i in range(n_units):
        g, members = units[i]
        unit_size = len(members)
        ul = unit_labels[i]

        # Nếu val đã đủ, gán tất cả còn lại vào train
        if val_count >= target_val_total:
            train_units.append((g, members))
            # Vẫn cập nhật remaining_total
            for c in range(n_classes):
                if ul[c]:
                    remaining_total[c] = max(0.0, remaining_total[c] - 1)
            continue

        # Tìm class c* có tỷ lệ cần val cao nhất
        active_classes = [c for c in range(n_classes) if ul[c] > 0]

        if not active_classes:
            # Ảnh không có nhãn hợp lệ — theo tỷ lệ tổng
            val_still_needed = target_val_total - val_count
            remaining_n = n - (sum(len(m) for _, m in val_units) + sum(len(m) for _, m in train_units))
            if val_still_needed / (remaining_n + 1e-9) > 0.5:
                val_units.append((g, members))
                val_count += unit_size
            else:
                train_units.append((g, members))
        else:
            # c* = class cần val nhất (tỷ lệ remaining_desired_val / remaining_total cao nhất)
            c_star = max(active_classes,
                         key=lambda c: remaining_desired_val[c] / (remaining_total[c] + 1e-9))

            # Xác suất gán vào val dựa trên nhu cầu của c*
            p_val = remaining_desired_val[c_star] / (remaining_total[c_star] + 1e-9)
            # Clamp để tránh > 1 do tích lũy sai số
            p_val = min(p_val, 1.0)

            if rng.random() < p_val:
                val_units.append((g, members))
                val_count += unit_size
                for c in range(n_classes):
                    if ul[c]:
                        remaining_desired_val[c] = max(0.0, remaining_desired_val[c] - 1)
            else:
                train_units.append((g, members))

        for c in range(n_classes):
            if ul[c]:
                remaining_total[c] = max(0.0, remaining_total[c] - 1)

    # Nếu val vẫn thiếu (do may rủi với seed), bổ sung từ train
    # Đây là trường hợp fallback để đảm bảo đúng kích thước
    val_images_flat = []
    for _, members in val_units:
        val_images_flat.extend(members)
    train_images_flat = []
    for _, members in train_units:
        train_images_flat.extend(members)

    # Nếu thiếu hoặc thừa, điều chỉnh bằng cách di chuyển ảnh đơn từ train sang val
    while len(val_images_flat) < target_val_total and len(train_images_flat) > 0:
        # Lấy ảnh cuối từ train (không thuộc near-dup group) vào val
        moved = False
        for idx in range(len(train_images_flat) - 1, -1, -1):
            img = train_images_flat[idx]
            # Chỉ di chuyển ảnh đơn lẻ (không trong near-dup group)
            if img not in group_map:
                val_images_flat.append(img)
                train_images_flat.pop(idx)
                moved = True
                break
        if not moved:
            # Phải di chuyển cả group
            img = train_images_flat.pop()
            val_images_flat.append(img)

    while len(val_images_flat) > target_val_total + 2:  # cho phép lệch tối đa 2
        img = val_images_flat.pop()
        train_images_flat.append(img)

    return sorted(train_images_flat), sorted(val_images_flat)
    """
    Chia train/val bằng thuật toán Iterative Stratification (Sechidis et al., 2011).

    Thuật toán:
      1. Tính số mẫu mong muốn của từng class trong val (target_val[c] = round(total_c * val_ratio)).
      2. Lặp cho đến khi hết ảnh:
         a. Tìm class c* có ít mẫu val còn thiếu nhất (nhưng > 0).
         b. Trong số ảnh chứa c*, chọn ảnh có ít nhãn nhất (ảnh "hiếm" hơn được ưu tiên).
         c. Gán ảnh đó vào val hoặc train dựa trên tỷ lệ cần đạt.
         d. Cập nhật target_val.

    Ràng buộc bổ sung: Các ảnh trong cùng nhóm near-dup được phân vào cùng một tập.
    Ưu tiên tránh leakage (cùng nhóm → cùng tập), ngay cả khi ảnh hưởng tỷ lệ class.
    """
    rng = random.Random(seed)
    n = len(image_names)
    n_classes = label_matrix.shape[1]
    target_val_total = round(n * val_ratio)

    # Tính số mẫu mong muốn từng class trong val
    class_totals = label_matrix.sum(axis=0)
    desired_val = np.round(class_totals * val_ratio).astype(int)

    # Xử lý nhóm near-dup: ảnh cùng nhóm phải vào cùng một tập
    # Nhóm hóa các ảnh
    group_to_images = defaultdict(list)
    for name in image_names:
        g = group_map.get(name, None)
        group_to_images[g].append(name)

    # Mỗi "đơn vị" chia là một group (None = nhóm đơn lẻ)
    units = []
    for g, members in group_to_images.items():
        units.append((g, sorted(members)))
    rng.shuffle(units)

    # Tạo ma trận multi-label theo đơn vị (group)
    # Một unit có nhãn c nếu ít nhất một ảnh trong unit có nhãn c
    unit_labels = []
    for g, members in units:
        combined = label_matrix[[image_names.index(m) for m in members]].max(axis=0)
        unit_labels.append(combined)
    unit_labels = np.array(unit_labels)

    # ----- Iterative Stratification (phiên bản xác định, không dùng random) -----
    # Thuật toán Sechidis et al. 2011 (phiên bản greedy tham lam thuần):
    #
    # Với mỗi ảnh chưa được gán:
    #   1. Tìm class c* có ít nhãn còn lại nhất trong tập chưa gán (tỷ lệ so với mong muốn).
    #      Nếu không có class nào (ảnh rỗng), dùng tỷ lệ tổng thể.
    #   2. Nếu ảnh không chứa c*: gán vào tập có tỷ lệ c* lớn hơn (để giữ cân bằng c*).
    #   3. Nếu ảnh chứa c*: gán vào val nếu còn chỗ cho val (remaining_desired_val[c*] > 0
    #      và tỷ lệ val/total > 0.5), ngược lại gán train.
    #   Điều chỉnh remaining_desired_val sau mỗi bước.

    remaining_desired_val = desired_val.copy().astype(float)
    # remaining_total[c] = số ảnh chứa c chưa được gán
    remaining_total = class_totals.copy().astype(float)

    val_units = []
    train_units = []
    val_count = 0
    train_count = 0
    n_units = len(units)

    # Sắp xếp units theo thứ tự cố định trước (không random, để tái hiện)
    # Dùng thứ tự sắp xếp theo tên ảnh đầu tiên trong unit
    units_sorted = sorted(units, key=lambda u: u[1][0])
    unit_labels_sorted = []
    for g, members in units_sorted:
        combined = label_matrix[[image_names.index(m) for m in members]].max(axis=0)
        unit_labels_sorted.append(combined)
    unit_labels_sorted = np.array(unit_labels_sorted)

    for idx in range(n_units):
        g, members = units_sorted[idx]
        unit_size = len(members)
        ul = unit_labels_sorted[idx]

        # Tính số còn lại cần gán cho val
        remaining_n = n - val_count - train_count

        # Tìm class c* theo tiêu chí: class chứa ít nhãn còn lại nhất (tỷ lệ remaining_desired_val / remaining_total)
        active_classes = [c for c in range(n_classes) if ul[c] > 0 and remaining_total[c] > 0]

        # Quyết định gán val hay train dựa trên tỷ lệ cần đạt
        val_still_needed = target_val_total - val_count
        train_still_needed = (n - target_val_total) - train_count

        if not active_classes:
            # Ảnh background hoặc không còn class nào cần — gán theo tỷ lệ tổng
            if val_still_needed > 0 and (val_still_needed / (remaining_n + 1e-9)) >= val_ratio:
                go_val = True
            else:
                go_val = False
        else:
            # c* = class hiếm nhất (remaining_desired_val[c] / remaining_total[c] cao nhất)
            c_star = max(active_classes, key=lambda c: remaining_desired_val[c] / (remaining_total[c] + 1e-9))

            val_desire = remaining_desired_val[c_star]
            total_c = remaining_total[c_star]

            # Nếu val còn thiếu class c* nhiều hơn → gán val
            # Nếu val đã đủ hoặc quá → gán train
            if val_still_needed <= 0:
                go_val = False
            elif train_still_needed <= 0:
                go_val = True
            else:
                # So sánh xác suất: nếu tỷ lệ cần val > tỷ lệ cần train → chọn val
                val_score = val_desire / (total_c + 1e-9)
                train_score = (total_c - val_desire) / (total_c + 1e-9)
                # Dùng seed-based tie-break
                rng_draw = (hash(str(members)) % 1000) / 1000.0  # Quyết định xác định dựa trên tên ảnh
                go_val = val_score > rng_draw

        if go_val:
            val_units.append((g, members))
            val_count += unit_size
            for c in range(n_classes):
                if ul[c]:
                    remaining_desired_val[c] = max(0, remaining_desired_val[c] - 1)
        else:
            train_units.append((g, members))
            train_count += unit_size

        # Cập nhật remaining_total
        for c in range(n_classes):
            if ul[c]:
                remaining_total[c] = max(0, remaining_total[c] - 1)

    # Giải phẳng
    val_images = []
    for _, members in val_units:
        val_images.extend(members)

    train_images = []
    for _, members in train_units:
        train_images.extend(members)
    return sorted(train_images), sorted(val_images)


def compute_split_stats(images: list, label_matrix: np.ndarray, image_names: list) -> dict:
    """Tính số object và số ảnh chứa class cho một tập."""
    indices = [image_names.index(img) for img in images]
    sub_matrix = label_matrix[indices]
    return {
        "image_count": len(images),
        "class_image_counts": sub_matrix.sum(axis=0),
    }


def main():
    data_raw = PROJECT_ROOT / "data" / "raw"
    train_images_dir = data_raw / "train" / "images"
    train_labels_dir = data_raw / "train" / "labels"
    classes_file = data_raw / "classes.txt"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    eda_dir = PROJECT_ROOT / "outputs" / "eda"
    processed_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 3A: CHIA DỮ LIỆU OBJECT DETECTION (TRAIN / VAL)")
    print("=" * 60)

    # 1. Đọc danh sách ảnh và classes
    classes = load_classes(classes_file)
    n_classes = len(classes)
    image_exts = {".png", ".jpg", ".jpeg"}
    image_files = sorted([f for f in train_images_dir.iterdir() if f.suffix.lower() in image_exts])
    image_names = [f.name for f in image_files]
    n = len(image_names)
    print(f"\n[1] Tổng số ảnh: {n} | Số classes: {n_classes}")

    # 2. Tạo ma trận multi-label và đếm object per image
    print("\n[2] Đang tạo ma trận multi-label và đếm object...")
    label_matrix = np.zeros((n, n_classes), dtype=np.int32)
    objects_per_image = np.zeros(n, dtype=np.int32)
    objects_per_image_per_class = np.zeros((n, n_classes), dtype=np.int32)

    for i, img_name in enumerate(image_names):
        stem = Path(img_name).stem
        label_path = train_labels_dir / f"{stem}.txt"
        annotations = load_yolo_label(label_path)
        for cid, *_ in annotations:
            if 0 <= cid < n_classes:
                label_matrix[i, cid] = 1
                objects_per_image_per_class[i, cid] += 1
        objects_per_image[i] = len(annotations)

    total_objects = int(objects_per_image.sum())
    class_totals = label_matrix.sum(axis=0)
    print(f"   Tổng object: {total_objects}")
    print(f"   Object/ảnh: mean={objects_per_image.mean():.2f}, min={objects_per_image.min()}, max={objects_per_image.max()}")

    # 3. Phát hiện ảnh gần trùng
    print("\n[3] Phát hiện ảnh gần trùng (pHash)...")
    try:
        group_map, pairs_found, near_dup_groups_dict = build_near_dup_groups(image_files)
    except ImportError:
        print("  [!] scipy không có sẵn, bỏ qua bước pHash. Dùng dict rỗng.")
        group_map, pairs_found, near_dup_groups_dict = {}, [], {}

    # Báo cáo rõ ràng về các nhóm gần trùng
    if near_dup_groups_dict:
        print(f"\n  [!] CẢNH BÁO: Phát hiện {len(near_dup_groups_dict)} nhóm ảnh gần trùng:")
        for gid, (_, members) in enumerate(near_dup_groups_dict.items()):
            print(f"       Nhóm {gid+1}: {members}")
        print("  Các ảnh cùng nhóm sẽ được đặt vào cùng một tập để tránh leakage.")
    else:
        print("  -> Không phát hiện ảnh gần trùng (pHash Hamming <= 10).")

    # 4. Thực hiện chia Iterative Stratification
    print("\n[4] Thực hiện Iterative Stratification split (val_ratio=20%)...")
    train_images, val_images = iterative_stratification_split(
        image_names, label_matrix, group_map, near_dup_groups_dict
    )
    print(f"   Train: {len(train_images)} ảnh | Val: {len(val_images)} ảnh")

    # 5. Kiểm tra kết quả cơ bản
    assert len(train_images) + len(val_images) == n, "Tổng số ảnh không khớp!"
    assert len(set(train_images) & set(val_images)) == 0, "Có ảnh xuất hiện ở cả hai tập!"
    print("   [+] Không có ảnh trùng lặp giữa train và val.")

    # Kiểm tra class coverage
    train_idx = [image_names.index(img) for img in train_images]
    val_idx = [image_names.index(img) for img in val_images]
    train_class_coverage = label_matrix[train_idx].max(axis=0)
    val_class_coverage = label_matrix[val_idx].max(axis=0)
    missing_in_train = [classes[c] for c in range(n_classes) if train_class_coverage[c] == 0]
    missing_in_val = [classes[c] for c in range(n_classes) if val_class_coverage[c] == 0]

    if missing_in_train:
        print(f"   [!] CẢNH BÁO: Các class không có trong TRAIN: {missing_in_train}")
    else:
        print(f"   [+] Tất cả {n_classes} class đều có trong train.")

    if missing_in_val:
        print(f"   [!] CẢNH BÁO: Các class không có trong VAL: {missing_in_val}")
    else:
        print(f"   [+] Tất cả {n_classes} class đều có trong val.")

    # 6. Thống kê phân bố class
    print("\n[5] Thống kê phân bố class:")
    train_class_imgs = label_matrix[train_idx].sum(axis=0)
    val_class_imgs = label_matrix[val_idx].sum(axis=0)
    train_class_objs = objects_per_image_per_class[train_idx].sum(axis=0)
    val_class_objs = objects_per_image_per_class[val_idx].sum(axis=0)

    print(f"\n  {'ID':<4} {'Class':<24} {'Train imgs':<12} {'Val imgs':<10} {'Train objs':<12} {'Val objs':<10} {'Val%':<8}")
    print("  " + "-" * 82)
    for cid in range(n_classes):
        ti = int(train_class_imgs[cid])
        vi = int(val_class_imgs[cid])
        to_ = int(train_class_objs[cid])
        vo = int(val_class_objs[cid])
        total_imgs = ti + vi
        pct = f"{vi/(total_imgs+1e-9)*100:.1f}%"
        print(f"  {cid:<4} {classes[cid]:<24} {ti:<12} {vi:<10} {to_:<12} {vo:<10} {pct:<8}")

    # 7. Kiểm tra ảnh gần trùng có bị rơi sang hai tập không
    leakage_warnings = []
    if near_dup_groups_dict:
        train_set = set(train_images)
        val_set = set(val_images)
        for gid, (_, members) in enumerate(near_dup_groups_dict.items()):
            in_train = [m for m in members if m in train_set]
            in_val = [m for m in members if m in val_set]
            if in_train and in_val:
                msg = f"Nhóm {gid+1} bị chia: train={in_train}, val={in_val}"
                leakage_warnings.append(msg)
                print(f"  [!] CẢNH BÁO LEAKAGE: {msg}")

    if not leakage_warnings and near_dup_groups_dict:
        print("  [+] Tất cả nhóm ảnh gần trùng đều nằm trong cùng một tập.")

    # 8. Lưu manifest files
    det_train_path = processed_dir / "detection_train.txt"
    det_val_path = processed_dir / "detection_val.txt"

    with open(det_train_path, "w", encoding="utf-8") as f:
        for img_name in train_images:
            f.write(img_name + "\n")

    with open(det_val_path, "w", encoding="utf-8") as f:
        for img_name in val_images:
            f.write(img_name + "\n")

    print(f"\n[+] Đã lưu: {det_train_path}")
    print(f"[+] Đã lưu: {det_val_path}")

    # 9. Lưu report CSV
    report_csv = eda_dir / "detection_split_report.csv"
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "train_images", "val_images", "train_objects", "val_objects"])
        for cid in range(n_classes):
            writer.writerow([
                cid,
                classes[cid],
                int(train_class_imgs[cid]),
                int(val_class_imgs[cid]),
                int(train_class_objs[cid]),
                int(val_class_objs[cid]),
            ])
    print(f"[+] Đã lưu: {report_csv}")

    # 10. In tóm tắt cuối
    print("\n" + "=" * 60)
    print("TÓM TẮT SPLIT DETECTION")
    print("=" * 60)
    print(f"Train: {len(train_images)} ảnh")
    print(f"Val:   {len(val_images)} ảnh")
    print(f"Leakage warnings: {len(leakage_warnings)}")
    if leakage_warnings:
        print("  CẢNH BÁO LEAKAGE (ảnh gần trùng bị chia split):")
        for w in leakage_warnings:
            print(f"    - {w}")
    else:
        print("  Không phát hiện leakage từ nhóm near-dup.")
    print("  Lưu ý: 'không phát hiện' dựa trên pHash Hamming <= 10, không loại trừ")
    print("  khả năng có ảnh cùng cảnh không bị phát hiện bởi thuật toán này.")


if __name__ == "__main__":
    main()
