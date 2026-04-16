"""
A-PAS: CARLA 데이터 검증 및 학습 준비
======================================
팀원에게 받은 carla_data/를 검증하고,
YOLO 학습용 폴더 구조로 변환합니다.

사용법:
  1. 팀원에게 받은 폴더를 carla_data/ 에 넣기
     carla_data/
     ├── images/
     │   ├── frame_000001.jpg
     │   └── ...
     └── labels/
         ├── frame_000001.txt
         └── ...

  2. 실행
     python prepare_yolo_data.py

  3. 문제 없으면 바로 학습
     python train_yolo_carla.py
"""

import os
import glob
import shutil
import random
import numpy as np
from collections import Counter

# ==========================================
# ⚙️ [설정]
# ==========================================
# 스크립트 위치 기준 절대경로 (어디서 실행해도 동작)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 입력: 팀원에게 받은 원본 데이터
INPUT_DIR = os.path.join(BASE_DIR, "carla_data")
IMG_DIR   = os.path.join(INPUT_DIR, "images")
LBL_DIR   = os.path.join(INPUT_DIR, "labels")

# 출력: YOLO 학습용 데이터셋
OUTPUT_DIR = os.path.join(BASE_DIR, "carla_dataset")

# Train/Val 비율
VAL_RATIO  = 0.1
RANDOM_SEED = 42

# A-PAS 클래스 정의
CLASS_NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
NUM_CLASSES = 5

# ==========================================
# 🔍 [Step 1] 데이터 검증
# ==========================================
def verify_data():
    print("=" * 55)
    print("🔍 [Step 1] 데이터 검증")
    print("=" * 55)

    errors = []
    warnings = []

    # 1-1. 폴더 존재 확인
    if not os.path.exists(IMG_DIR):
        errors.append(f"❌ 이미지 폴더 없음: {IMG_DIR}")
        return errors, warnings
    if not os.path.exists(LBL_DIR):
        errors.append(f"❌ 라벨 폴더 없음: {LBL_DIR}")
        return errors, warnings

    # 1-2. 파일 수집
    img_files = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMG_DIR, "*.png")) +
        glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    )
    lbl_files = sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))

    img_names = {os.path.splitext(os.path.basename(f))[0] for f in img_files}
    lbl_names = {os.path.splitext(os.path.basename(f))[0] for f in lbl_files}

    print(f"  이미지: {len(img_files)}장")
    print(f"  라벨  : {len(lbl_files)}개")

    # 1-3. 이미지-라벨 매칭 확인
    imgs_without_labels = img_names - lbl_names
    labels_without_imgs = lbl_names - img_names
    matched = img_names & lbl_names

    if imgs_without_labels:
        warnings.append(
            f"⚠️  라벨 없는 이미지 {len(imgs_without_labels)}장 "
            f"(학습에서 제외됨)"
        )
    if labels_without_imgs:
        warnings.append(
            f"⚠️  이미지 없는 라벨 {len(labels_without_imgs)}개 "
            f"(무시됨)"
        )

    print(f"  매칭됨: {len(matched)}쌍")

    if len(matched) == 0:
        errors.append("❌ 매칭되는 이미지-라벨 쌍이 0개입니다!")
        return errors, warnings

    if len(matched) < 100:
        warnings.append(
            f"⚠️  데이터가 {len(matched)}장으로 적습니다. "
            f"500장 이상 권장"
        )

    # 1-4. 라벨 내용 검증
    print("\n  라벨 내용 검증 중...")
    class_counts = Counter()
    bad_labels = []
    total_objects = 0
    person_images = 0

    for name in sorted(matched):
        lbl_path = os.path.join(LBL_DIR, f"{name}.txt")
        has_person = False

        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # 형식 확인: cls x y w h (5개 값)
            if len(parts) != 5:
                bad_labels.append(
                    f"{name}.txt 줄{line_num}: "
                    f"값 {len(parts)}개 (5개여야 함)"
                )
                continue

            try:
                cls_id = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), \
                             float(parts[3]), float(parts[4])
            except ValueError:
                bad_labels.append(
                    f"{name}.txt 줄{line_num}: 숫자 변환 오류"
                )
                continue

            # 클래스 번호 확인
            if cls_id not in CLASS_NAMES:
                bad_labels.append(
                    f"{name}.txt 줄{line_num}: "
                    f"클래스 {cls_id} (0~4만 유효)"
                )
                continue

            # 좌표 범위 확인 (0~1 정규화)
            for val, val_name in [(x, "x"), (y, "y"), (w, "w"), (h, "h")]:
                if val < 0 or val > 1:
                    bad_labels.append(
                        f"{name}.txt 줄{line_num}: "
                        f"{val_name}={val:.4f} (0~1 범위 벗어남 → 정규화 안 됨!)"
                    )

            class_counts[cls_id] += 1
            total_objects += 1

            if cls_id == 0:
                has_person = True

        if has_person:
            person_images += 1

    # 1-5. 결과 출력
    print(f"\n  📊 클래스 분포:")
    for cls_id in range(NUM_CLASSES):
        count = class_counts.get(cls_id, 0)
        bar = "█" * min(30, count // max(1, total_objects // 30))
        pct = count / max(1, total_objects) * 100
        print(f"    {cls_id} {CLASS_NAMES[cls_id]:12s}: "
              f"{count:>5,}개 ({pct:5.1f}%) {bar}")

    print(f"\n  총 객체 수      : {total_objects:,}개")
    print(f"  이미지당 평균   : {total_objects / max(1, len(matched)):.1f}개")

    person_ratio = person_images / max(1, len(matched)) * 100
    print(f"  Person 포함 비율: {person_ratio:.1f}% "
          f"({person_images}/{len(matched)})")

    # 경고/에러 수집
    if bad_labels:
        if len(bad_labels) > 10:
            errors.append(
                f"❌ 잘못된 라벨 {len(bad_labels)}개 발견! "
                f"(처음 5개: {bad_labels[:5]})"
            )
        else:
            for bl in bad_labels:
                errors.append(f"❌ {bl}")

    if class_counts.get(0, 0) == 0:
        errors.append("❌ Person(0) 클래스가 하나도 없습니다!")

    if person_ratio < 30:
        warnings.append(
            f"⚠️  Person 포함 이미지가 {person_ratio:.0f}%로 적습니다. "
            f"50% 이상 권장"
        )

    # 1920이나 1080 같은 값이 있으면 정규화 안 된 것
    if any("정규화 안 됨" in bl for bl in bad_labels):
        errors.append(
            "❌ 좌표가 0~1로 정규화되지 않았습니다! "
            "팀원에게 이미지 크기로 나눠달라고 요청하세요."
        )

    return errors, warnings


# ==========================================
# 📦 [Step 2] Train/Val 분할 및 폴더 구성
# ==========================================
def prepare_dataset():
    print("\n" + "=" * 55)
    print("📦 [Step 2] Train/Val 분할")
    print("=" * 55)

    # 매칭된 파일만 수집
    img_files = sorted(
        glob.glob(os.path.join(IMG_DIR, "*.jpg")) +
        glob.glob(os.path.join(IMG_DIR, "*.png")) +
        glob.glob(os.path.join(IMG_DIR, "*.jpeg"))
    )

    matched_files = []
    for img_path in img_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(LBL_DIR, f"{name}.txt")
        if os.path.exists(lbl_path):
            matched_files.append((img_path, lbl_path))

    # 셔플 후 분할
    random.seed(RANDOM_SEED)
    random.shuffle(matched_files)

    val_count = max(1, int(len(matched_files) * VAL_RATIO))
    val_set   = matched_files[:val_count]
    train_set = matched_files[val_count:]

    print(f"  전체: {len(matched_files)}쌍")
    print(f"  Train: {len(train_set)}쌍 ({100 - VAL_RATIO*100:.0f}%)")
    print(f"  Val  : {len(val_set)}쌍 ({VAL_RATIO*100:.0f}%)")

    # 폴더 생성 및 복사
    for split, file_set in [("train", train_set), ("val", val_set)]:
        img_dst = os.path.join(OUTPUT_DIR, "images", split)
        lbl_dst = os.path.join(OUTPUT_DIR, "labels", split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for img_path, lbl_path in file_set:
            shutil.copy2(img_path, img_dst)
            shutil.copy2(lbl_path, lbl_dst)

    print(f"  ✅ {OUTPUT_DIR}/ 폴더 구성 완료")


# ==========================================
# 📝 [Step 3] data.yaml 생성
# ==========================================
def create_data_yaml():
    print("\n" + "=" * 55)
    print("📝 [Step 3] data.yaml 생성")
    print("=" * 55)

    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    abs_path = os.path.abspath(OUTPUT_DIR)

    yaml_content = f"""# A-PAS CARLA Dataset for YOLOv8 Fine-tuning
# Auto-generated by prepare_yolo_data.py

path: {abs_path}
train: images/train
val: images/val

nc: {NUM_CLASSES}
names:
  0: person
  1: car
  2: motorcycle
  3: bus
  4: truck
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  ✅ {yaml_path} 생성 완료")


# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  A-PAS YOLO Fine-tuning 데이터 준비 도구    ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    # Step 1: 검증
    errors, warnings = verify_data()

    # 경고 출력
    if warnings:
        print(f"\n{'─' * 55}")
        print(f"⚠️  경고 {len(warnings)}개:")
        for w in warnings:
            print(f"  {w}")

    # 에러 출력 → 에러 있으면 중단
    if errors:
        print(f"\n{'─' * 55}")
        print(f"❌ 에러 {len(errors)}개 — 수정 후 다시 실행하세요:")
        for e in errors:
            print(f"  {e}")
        print(f"\n💡 팀원에게 데이터 수정 요청 후 다시 받으세요.")
        return

    print(f"\n✅ 검증 통과!")

    # Step 2: Train/Val 분할
    prepare_dataset()

    # Step 3: data.yaml 생성
    create_data_yaml()

    # 완료
    print("\n" + "=" * 55)
    print("🎉 준비 완료! 바로 학습 가능")
    print("=" * 55)
    print()
    print("  다음 명령어 실행:")
    print("  python train_yolo_carla.py")
    print()


if __name__ == "__main__":
    main()