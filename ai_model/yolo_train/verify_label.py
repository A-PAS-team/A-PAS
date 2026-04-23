"""
A-PAS: CARLA 라벨 검증 및 보완 도구
=====================================
1. 시각적 검증: 이미지 위에 라벨 bbox를 그려서 확인
2. 누락 탐지: YOLOv8s로 추가 탐지 후 놓친 객체 보완
3. 통계 리포트: 클래스별 분포, 이미지당 객체 수 등

사용법:
  # 1) 시각적 검증만 (라벨이 맞는지 눈으로 확인)
  python verify_labels.py --mode visual

  # 2) 누락 보완 (YOLOv8s로 놓친 객체 추가)
  python verify_labels.py --mode fix

  # 3) 둘 다
  python verify_labels.py --mode all
"""

import os
import sys
import glob
import cv2
import numpy as np
import argparse
from collections import Counter

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "carla_data_v2")
IMG_DIR   = os.path.join(INPUT_DIR, "images")
LBL_DIR   = os.path.join(INPUT_DIR, "labels")

# 시각적 검증 출력
VISUAL_DIR = os.path.join(BASE_DIR, "label_check")

# 클래스 정의
CLASS_NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
CLASS_COLORS = {
    0: (0, 0, 255),     # person: 빨강
    1: (0, 255, 0),     # car: 초록
    2: (255, 0, 255),   # motorcycle: 핑크
    3: (255, 165, 0),   # bus: 주황
    4: (255, 255, 0),   # truck: 노랑
}

# Pseudo-labeling 보완 설정
YOLO_MODEL  = "yolov8s.pt"
YOLO_IMGSZ  = 1280
YOLO_CONF   = 0.15
IOU_THRESHOLD = 0.3  # 기존 라벨과 이 이상 겹치면 중복으로 판단

# COCO → A-PAS 클래스 매핑
COCO_TO_APAS = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}


# ==========================================
# 🔧 [유틸 함수]
# ==========================================
def load_labels(lbl_path):
    """라벨 파일 로드 → [(cls, x_c, y_c, w, h), ...]"""
    labels = []
    if not os.path.exists(lbl_path):
        return labels
    with open(lbl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), \
                             float(parts[3]), float(parts[4])
                labels.append((cls, x, y, w, h))
            except ValueError:
                continue
    return labels


def save_labels(lbl_path, labels):
    """라벨 리스트를 파일로 저장"""
    with open(lbl_path, "w") as f:
        for cls, x, y, w, h in labels:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_pixel(x_c, y_c, w, h, img_w, img_h):
    """YOLO format → 픽셀 좌표 (x1, y1, x2, y2)"""
    x1 = int((x_c - w / 2) * img_w)
    y1 = int((y_c - h / 2) * img_h)
    x2 = int((x_c + w / 2) * img_w)
    y2 = int((y_c + h / 2) * img_h)
    return x1, y1, x2, y2


def calc_iou(box1, box2):
    """두 박스의 IoU 계산 (YOLO format: x_c, y_c, w, h)"""
    # YOLO → (x1, y1, x2, y2)
    ax1 = box1[0] - box1[2] / 2
    ay1 = box1[1] - box1[3] / 2
    ax2 = box1[0] + box1[2] / 2
    ay2 = box1[1] + box1[3] / 2

    bx1 = box2[0] - box2[2] / 2
    by1 = box2[1] - box2[3] / 2
    bx2 = box2[0] + box2[2] / 2
    by2 = box2[1] + box2[3] / 2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / max(union, 1e-6)


# ==========================================
# 👁️ [시각적 검증]
# ==========================================
def visual_verify():
    print("\n" + "=" * 55)
    print("👁️  시각적 검증 — 라벨을 이미지 위에 시각화")
    print("=" * 55)

    os.makedirs(VISUAL_DIR, exist_ok=True)

    img_files = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMG_DIR, "*.png"))
    )

    if not img_files:
        print(f"  ❌ {IMG_DIR}에 이미지가 없습니다!")
        return

    # 최대 50장만 시각화 (전체 다 하면 느림)
    sample = img_files[:50]
    stats = {"empty": 0, "person_only": 0, "car_only": 0, "mixed": 0}

    for img_path in sample:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, f"{name}.txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        labels = load_labels(lbl_path)

        if not labels:
            stats["empty"] += 1

        has_person = False
        has_vehicle = False

        for cls, x_c, y_c, w, h in labels:
            x1, y1, x2, y2 = yolo_to_pixel(x_c, y_c, w, h, img_w, img_h)
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            label_text = CLASS_NAMES.get(cls, f"cls{cls}")

            # bbox 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # 라벨 텍스트
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, 2)[0]
            cv2.rectangle(img, (x1, y1 - text_size[1] - 8),
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # 박스 크기 표시 (px)
            box_w_px = x2 - x1
            box_h_px = y2 - y1
            cv2.putText(img, f"{box_w_px}x{box_h_px}",
                       (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            if cls == 0:
                has_person = True
            else:
                has_vehicle = True

        if has_person and has_vehicle:
            stats["mixed"] += 1
        elif has_person:
            stats["person_only"] += 1
        elif has_vehicle:
            stats["car_only"] += 1

        # 저장
        cv2.imwrite(os.path.join(VISUAL_DIR, f"{name}_check.jpg"), img)

    print(f"  ✅ {len(sample)}장 시각화 완료 → {VISUAL_DIR}/")
    print(f"\n  분포 (샘플 {len(sample)}장 기준):")
    print(f"    사람+차량 같이: {stats['mixed']}장")
    print(f"    사람만       : {stats['person_only']}장")
    print(f"    차량만       : {stats['car_only']}장")
    print(f"    빈 라벨      : {stats['empty']}장")
    print(f"\n  💡 {VISUAL_DIR}/ 폴더에서 이미지를 열어 확인하세요!")
    print(f"     빨간 박스=person, 초록 박스=car")


# ==========================================
# 🔧 [누락 보완] — YOLOv8s로 놓친 객체 추가
# ==========================================
def fix_missing_labels():
    print("\n" + "=" * 55)
    print("🔧 누락 보완 — YOLOv8s로 놓친 객체 탐지 후 추가")
    print("=" * 55)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ❌ ultralytics 설치 필요: pip install ultralytics")
        return

    img_files = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMG_DIR, "*.png"))
    )

    if not img_files:
        print(f"  ❌ {IMG_DIR}에 이미지가 없습니다!")
        return

    print(f"  🧠 {YOLO_MODEL} 로드 중 (imgsz={YOLO_IMGSZ}, conf={YOLO_CONF})...")
    model = YOLO(YOLO_MODEL)

    total_added = 0
    total_images_fixed = 0
    added_by_class = Counter()

    # 백업 폴더
    backup_dir = os.path.join(INPUT_DIR, "labels_backup")
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir, exist_ok=True)
        # 기존 라벨 백업
        for f in glob.glob(os.path.join(LBL_DIR, "*.txt")):
            import shutil
            shutil.copy2(f, backup_dir)
        print(f"  📁 기존 라벨 백업 완료 → {backup_dir}/")

    for idx, img_path in enumerate(img_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, f"{name}.txt")

        # 기존 라벨 로드
        existing_labels = load_labels(lbl_path)
        existing_boxes = [(x, y, w, h) for (_, x, y, w, h) in existing_labels]

        # YOLO 추론
        results = model(img_path, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

        new_labels = list(existing_labels)  # 기존 라벨 복사
        added_count = 0

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                coco_cls = int(boxes.cls[i].item())

                # A-PAS 클래스만
                if coco_cls not in COCO_TO_APAS:
                    continue

                apas_cls = COCO_TO_APAS[coco_cls]
                xywhn = boxes.xywhn[i].cpu().numpy()
                x_c, y_c, w, h = xywhn[0], xywhn[1], xywhn[2], xywhn[3]

                # 기존 라벨과 중복 체크 (IoU 기반)
                is_duplicate = False
                for ex_box in existing_boxes:
                    iou = calc_iou((x_c, y_c, w, h), ex_box)
                    if iou > IOU_THRESHOLD:
                        is_duplicate = True
                        break

                # 중복 아니면 추가
                if not is_duplicate:
                    new_labels.append((apas_cls, x_c, y_c, w, h))
                    added_count += 1
                    added_by_class[apas_cls] += 1

        # 추가된 게 있으면 저장
        if added_count > 0:
            save_labels(lbl_path, new_labels)
            total_added += added_count
            total_images_fixed += 1

        # 진행 상황
        if (idx + 1) % 100 == 0 or idx == len(img_files) - 1:
            print(f"  [{idx+1}/{len(img_files)}] "
                  f"수정: {total_images_fixed}장 | 추가: {total_added}개")

    # 결과
    print(f"\n  📊 보완 결과:")
    print(f"    수정된 이미지  : {total_images_fixed}장")
    print(f"    추가된 객체    : {total_added}개")
    if added_by_class:
        print(f"    클래스별 추가:")
        for cls_id, count in sorted(added_by_class.items()):
            print(f"      {CLASS_NAMES[cls_id]:12s}: +{count}개")
    print(f"\n    기존 라벨 백업 : {backup_dir}/")

    if total_added == 0:
        print(f"\n  ✅ 누락 없음! 라벨이 잘 되어있습니다.")
    else:
        print(f"\n  ⚠️  {total_added}개 객체가 추가되었습니다.")
        print(f"     visual 모드로 다시 확인하는 걸 추천해요!")


# ==========================================
# 📊 [통계 리포트]
# ==========================================
def print_stats():
    print("\n" + "=" * 55)
    print("📊 데이터 통계 리포트")
    print("=" * 55)

    img_files = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMG_DIR, "*.png"))
    )
    lbl_files = sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))

    print(f"  이미지: {len(img_files)}장 | 라벨: {len(lbl_files)}개")

    class_counts = Counter()
    objects_per_image = []
    person_images = 0
    empty_labels = 0
    small_objects = 0  # 너무 작은 bbox

    for img_path in img_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, f"{name}.txt")
        labels = load_labels(lbl_path)

        objects_per_image.append(len(labels))
        if not labels:
            empty_labels += 1
            continue

        has_person = False
        for cls, x, y, w, h in labels:
            class_counts[cls] += 1
            if cls == 0:
                has_person = True
            # 너무 작은 객체 카운트 (3% 미만)
            if w < 0.03 or h < 0.03:
                small_objects += 1

        if has_person:
            person_images += 1

    total_obj = sum(class_counts.values())
    print(f"\n  클래스별 분포:")
    for cls_id in range(5):
        count = class_counts.get(cls_id, 0)
        pct = count / max(1, total_obj) * 100
        bar = "█" * min(30, int(pct))
        print(f"    {cls_id} {CLASS_NAMES[cls_id]:12s}: "
              f"{count:>5,}개 ({pct:5.1f}%) {bar}")

    print(f"\n  총 객체: {total_obj:,}개")
    if objects_per_image:
        arr = np.array(objects_per_image)
        print(f"  이미지당 객체: 평균 {arr.mean():.1f} / "
              f"최소 {arr.min()} / 최대 {arr.max()}")
    print(f"  빈 라벨 이미지: {empty_labels}장")
    print(f"  작은 객체 (3%미만): {small_objects}개")
    print(f"  Person 포함 이미지: {person_images}장 "
          f"({person_images/max(1,len(img_files))*100:.1f}%)")


# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="A-PAS CARLA 라벨 검증 및 보완"
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "fix", "stats", "all"],
        default="stats",
        help="visual: 시각화 검증 | fix: 누락 보완 | stats: 통계 | all: 전부"
    )
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  A-PAS CARLA 라벨 검증 및 보완 도구         ║")
    print("╚══════════════════════════════════════════════╝")

    if args.mode in ["stats", "all"]:
        print_stats()

    if args.mode in ["visual", "all"]:
        visual_verify()

    if args.mode in ["fix", "all"]:
        fix_missing_labels()

    print("\n🎉 완료!")


if __name__ == "__main__":
    main()