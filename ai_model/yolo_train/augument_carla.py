import os
import cv2
import numpy as np
import albumentations as A
import random
import glob
from tqdm import tqdm
from collections import Counter

# ==========================================
# ⚙️ 설정 및 하이퍼파라미터
# ==========================================
DATASET_PATH = r"c:\Users\USER\Desktop\UIJIN_Workfolder\workplace\A-PAS\ai_model\yolo_train\carla_dataset_v5"
TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "images", "train")
TRAIN_LBL_DIR = os.path.join(DATASET_PATH, "labels", "train")

# 클래스 매핑 (실제 클래스 ID에 맞게 수정 필요)
CLASS_NAMES = {
    0: "person",
    1: "car",
    2: "motorcycle",
    3: "large_vehicle"
}

# 클래스별 목표 증강 배율
AUG_MULTIPLIERS = {
    0: 1.5, # person: 1.5배 증강
    1: 0.0, # car: 증강 안 함
    2: 4.5, # motorcycle: 4~5배 증강
    3: 2.5  # large_vehicle: 2~3배 증강
}

# ==========================================
# 🎨 Albumentations 증강 파이프라인
# ==========================================
def get_train_transforms():
    return A.Compose(
        [
            A.Affine(scale=(0.5, 1.5), p=0.4),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.4, hue=0.15, p=0.7),
            A.OneOf([
                A.RandomRain(p=1.0),
                A.RandomFog(p=1.0),
                A.ToGray(p=1.0)
            ], p=0.3),
            A.OneOf([
                # 💡 피드백 반영 1: GaussNoise API 변경 대응
                # 구버전 기준. 만약 실행 시 에러나 Warning이 뜬다면 아래 줄을 주석 처리하고 그 밑의 코드를 활성화하세요.
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                # A.GaussNoise(std_range=(0.1, 0.3), p=1.0), # (최신 버전용 대체 코드)
                
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=7, p=1.0)
            ], p=0.4),
            A.CoarseDropout(
                max_holes=8, max_height=64, max_width=64, 
                min_holes=1, min_height=8, min_width=8, 
                fill_value=0, p=0.5
            )
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    )

# ==========================================
# 📊 유틸리티 함수
# ==========================================
def count_classes(label_dir):
    counter = Counter()
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    for file in label_files:
        with open(file, 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counter[class_id] += 1
    return counter

def calculate_augment_count(class_ids):
    if not class_ids:
        return 0
    
    max_multiplier = max([AUG_MULTIPLIERS.get(c, 0) for c in class_ids])
    
    if max_multiplier == 0:
        return 0
        
    base_count = int(max_multiplier)
    prob = max_multiplier - base_count
    
    if random.random() < prob:
        return base_count + 1
    return base_count

# ==========================================
# 🚀 메인 실행 로직
# ==========================================
def main():
    print("=== 📊 증강 전 데이터셋 분포 ===")
    before_counts = count_classes(TRAIN_LBL_DIR)
    for cid, count in sorted(before_counts.items()):
        print(f" - {CLASS_NAMES.get(cid, f'Class {cid}')}: {count}개")
        
    transform = get_train_transforms()
    
    # 💡 피드백 반영 2: jpg뿐만 아니라 png, jpeg 확장자까지 모두 수집
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(glob.glob(os.path.join(TRAIN_IMG_DIR, ext)))
        
    print(f"\n총 {len(image_files)}개의 학습 이미지를 검사합니다...")
    
    aug_created = 0
    
    for img_path in tqdm(image_files, desc="데이터 증강 진행 중"):
        filename = os.path.basename(img_path)
        basename, ext = os.path.splitext(filename)
        
        if "_aug_" in basename:
            continue
            
        lbl_path = os.path.join(TRAIN_LBL_DIR, basename + ".txt")
        if not os.path.exists(lbl_path):
            continue
            
        bboxes = []
        class_labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    c_id, x, y, w, h = map(float, parts[:5])
                    class_labels.append(int(c_id))
                    bboxes.append([x, y, w, h])
                    
        unique_classes = set(class_labels)
        num_augs = calculate_augment_count(unique_classes)
        
        if num_augs <= 0:
            continue
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for i in range(num_augs):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_image = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_classes = transformed['class_labels']
                
                if len(aug_bboxes) > 0:
                    new_basename = f"{basename}_aug_{i}"
                    
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(TRAIN_IMG_DIR, new_basename + ext), aug_image_bgr)
                    
                    with open(os.path.join(TRAIN_LBL_DIR, new_basename + ".txt"), 'w') as f:
                        for bbox, cls in zip(aug_bboxes, aug_classes):
                            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                            
                    aug_created += 1
            except Exception as e:
                pass

    print(f"\n✅ 증강 완료! 총 {aug_created}개의 증강된 이미지/라벨 쌍이 추가되었습니다.")
    
    print("\n=== 📈 증강 후 데이터셋 분포 ===")
    after_counts = count_classes(TRAIN_LBL_DIR)
    for cid, count in sorted(after_counts.items()):
        increase = after_counts[cid] - before_counts.get(cid, 0)
        print(f" - {CLASS_NAMES.get(cid, f'Class {cid}')}: {count}개 (+{increase}개 증가)")

if __name__ == "__main__":
    main()