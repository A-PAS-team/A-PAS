import cv2
import torch
import numpy as np
import os
import glob
import pandas as pd
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 프로젝트 규격 및 경로
# ==========================================
INPUT_BASE = r"E:\raw_data"        # 최상위 경로 (Training/Validation 폴더가 있는 곳)
SEQ_LENGTH, PRED_LENGTH = 20, 10   # 2초 관찰, 1초 예측
TOTAL_REQ = SEQ_LENGTH + PRED_LENGTH
DT = 0.1                           # 10FPS 기준

IMG_W, IMG_H = 1920, 1080
DIAG = np.sqrt(IMG_W**2 + IMG_H**2)

MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.3
CLASS_MAP = {0:0, 2:1, 5:2, 7:3, 3:4} # 사람, 자동차, 버스, 트럭, 오토바이

def process_and_extract(folder_path, model):
    """하위 폴더 내 모든 이미지를 ByteTrack으로 추적하여 17개 피처 추출"""
    # 재귀적으로 모든 하위 폴더의 jpg/png 탐색
    img_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True))
    if not img_files:
        img_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True))
    
    if not img_files: return None, None

    raw_list = []
    for idx, img_path in enumerate(img_files):
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        # ByteTrack 적용
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().tolist()
            cids = results[0].boxes.cls.int().cpu().tolist()
            
            frame_time = idx * DT # Timestamp 적용
            
            for box, tid, cid in zip(boxes, tids, cids):
                if cid in CLASS_MAP:
                    raw_list.append([frame_time, tid, cid, box[0], box[1], box[2], box[3]])

    if not raw_list: return None, None
    
    # 17개 피처 엔지니어링 로직
    df = pd.DataFrame(raw_list, columns=['time', 'tid', 'cid', 'x', 'y', 'w', 'h'])
    folder_X, folder_y = [], []

    for tid, group in df.groupby('tid'):
        group = group.sort_values('time')
        if len(group) < TOTAL_REQ: continue

        pos, size = group[['x', 'y']].values, group[['w', 'h']].values
        vel = np.gradient(pos, DT, axis=0)
        acc = np.gradient(vel, DT, axis=0)
        speed = np.sqrt(np.sum(vel**2, axis=1)).reshape(-1, 1)
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        
        # 물리 피처 12개 결합
        physics = np.hstack([
            pos[:,0:1]/IMG_W, pos[:,1:2]/IMG_H,
            vel[:,0:1]/IMG_W, vel[:,1:2]/IMG_H,
            acc[:,0:1]/IMG_W, acc[:,1:2]/IMG_H,
            speed/DIAG, np.sin(heading).reshape(-1,1), np.cos(heading).reshape(-1,1),
            size[:,0:1]/IMG_W, size[:,1:2]/IMG_H,
            (size[:,0]*size[:,1]).reshape(-1,1)/(IMG_W*IMG_H)
        ])

        # 클래스 원-핫 5개 결합
        cid = group['cid'].iloc[0]
        one_hot = np.zeros(5); one_hot[CLASS_MAP[cid]] = 1
        final_f = np.hstack([physics, np.tile(one_hot, (len(physics), 1))]).astype(np.float32)

        for i in range(len(final_f) - TOTAL_REQ + 1):
            folder_X.append(final_f[i : i + SEQ_LENGTH])
            folder_y.append(final_f[i + SEQ_LENGTH : i + TOTAL_REQ, :2])

    return folder_X, folder_y

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 A-PAS 통합 데이터 추출 (Training/Validation 분리 모드)")
    model = YOLO(MODEL_NAME).to(device)

    # 각 세트별로 루프를 돌며 데이터 수집 및 저장
    for split in ["Training", "Validation"]:
        split_path = os.path.join(INPUT_BASE, split)
        if not os.path.exists(split_path):
            print(f"⏭️  {split_path} 없음, 스킵합니다.")
            continue
            
        folders = [f for f in os.scandir(split_path) if f.is_dir()]
        print(f"\n📂 [{split}] 총 {len(folders)}개 폴더 처리 시작...")
        
        split_X, split_y = [], []
        for idx, folder in enumerate(folders):
            print(f"   [{idx+1}/{len(folders)}] {folder.name}...", end=' ')
            fx, fy = process_and_extract(folder.path, model)
            if fx:
                split_X.extend(fx)
                split_y.extend(fy)
                print(f"✅ ({len(fx)}개)")
            else:
                print("⚠️ 스킵")

        # 각 분기별로 별도 저장 (X_Training_17f.npy 등)
        if split_X:
            np.save(f"X_{split}_17f.npy", np.array(split_X, dtype=np.float32))
            np.save(f"y_{split}_17f.npy", np.array(split_y, dtype=np.float32))
            print(f"💾 [{split}] 저장 완료: {len(split_X)}개 시퀀스")

if __name__ == "__main__":
    main()