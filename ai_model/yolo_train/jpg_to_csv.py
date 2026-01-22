import cv2
import torch
import csv
import os
import glob
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 경로 및 모델 설정
# ==========================================
# 1. 최상위 데이터 경로 (SSD)
INPUT_BASE = r"D:\raw_data" 

# 2. 결과 CSV가 저장될 폴더 (A-PAS/ai_model/yolo_train/data)
OUTPUT_BASE = "data"

# 3. 사용할 YOLO 모델 및 설정
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.3
TARGET_CLASSES = [0, 1, 2, 3, 5, 7] # 사람, 자전거, 자동차, 오토바이, 버스, 트럭
# ==========================================

def process_sequence(folder_path, output_csv, model):
    """폴더 내 이미지를 처리하여 CSV로 저장"""
    img_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))

    if not img_files:
        return False

    with open(output_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'track_id', 'class_id', 'x_center', 'y_center', 'width', 'height'])

        frame_count = 0
        for img_path in img_files:
            frame = cv2.imread(img_path)
            if frame is None: continue

            frame_count += 1
            # persist=True로 객체 ID 추적 유지
            results = model.track(frame, persist=True, verbose=False, conf=CONF_THRESHOLD)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    if class_id in TARGET_CLASSES:
                        x, y, w, h = box
                        wr.writerow([frame_count, track_id, class_id, round(x, 2), round(y, 2), round(w, 2), round(h, 2)])
    return True

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 YOLOv8 데이터 추출 시작 (Device: {device})")
    model = YOLO(MODEL_NAME).to(device)

    # Training과 Validation 폴더 각각 처리
    for split in ["Training", "Validation"]:
        split_path = os.path.join(INPUT_BASE, split)
        if not os.path.exists(split_path):
            print(f"⏭️  {split} 폴더가 없어 건너뜁니다.")
            continue

        # 저장 경로 생성: data/Training, data/Validation
        save_dir = os.path.join(OUTPUT_BASE, split)
        os.makedirs(save_dir, exist_ok=True)

        # 해당 폴더 내의 모든 시퀀스 폴더 탐색
        folders = [f for f in os.scandir(split_path) if f.is_dir()]
        print(f"\n📂 [{split}] 총 {len(folders)}개 폴더 발견")

        for idx, folder in enumerate(folders):
            csv_name = f"normal_data_{folder.name}.csv"
            save_path = os.path.join(save_dir, csv_name)
            
            print(f"   [{idx+1}/{len(folders)}] 처리 중: {folder.name}...", end=' ')
            
            success = process_sequence(folder.path, save_path, model)
            if success:
                print("✅ 완료")
            else:
                print("⚠️ 이미지 없음 (패스)")

    print("\n🎉 모든 데이터 변환 완료!")

if __name__ == "__main__":
    main()