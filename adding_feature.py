import cv2
import torch
import numpy as np
import os
import glob
import pandas as pd
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 프로젝트 규격 고정
# ==========================================
INPUT_BASE = r"E:\raw_data"        # Training/Validation 폴더가 있는 곳
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.3
SEQ_LENGTH, PRED_LENGTH = 20, 10   # 2초 관찰, 1초 예측
TOTAL_REQ = SEQ_LENGTH + PRED_LENGTH
DT = 0.1                           # 10FPS

IMG_W, IMG_H = 1920, 1080
DIAG = np.sqrt(IMG_W**2 + IMG_H**2)
CLASS_MAP = {0:0, 2:1, 5:2, 7:3, 3:4} # 사람, 자동차, 버스, 트럭, 오토바이

# ==========================================
# 🛠️ [함수] 유틸리티 및 피처 엔지니어링
# ==========================================
def get_folder_num(folder_name):
    """폴더명 끝의 번호 추출 (예: ...C0189 -> 189)"""
    try:
        return int(folder_name.split('_')[-1][1:])
    except:
        return -1

def extract_physics_features(raw_list):
    """수집된 raw 데이터를 바탕으로 17개 피처 생성"""
    df = pd.DataFrame(raw_list, columns=['time', 'tid', 'cid', 'x', 'y', 'w', 'h'])
    X_list, y_list = [], []

    for tid, group in df.groupby('tid'):
        group = group.sort_values('time')
        if len(group) < TOTAL_REQ: continue

        pos, size = group[['x', 'y']].values, group[['w', 'h']].values
        vel = np.gradient(pos, DT, axis=0)
        acc = np.gradient(vel, DT, axis=0)
        speed = np.sqrt(np.sum(vel**2, axis=1)).reshape(-1, 1)
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        
        # 12개 물리 피처 결합
        physics = np.hstack([
            pos[:,0:1]/IMG_W, pos[:,1:2]/IMG_H,
            vel[:,0:1]/IMG_W, vel[:,1:2]/IMG_H,
            acc[:,0:1]/IMG_W, acc[:,1:2]/IMG_H,
            speed/DIAG, np.sin(heading).reshape(-1,1), np.cos(heading).reshape(-1,1),
            size[:,0:1]/IMG_W, size[:,1:2]/IMG_H,
            (size[:,0]*size[:,1]).reshape(-1,1)/(IMG_W*IMG_H)
        ])

        # 5개 클래스 피처 (One-hot)
        cid = group['cid'].iloc[0]
        one_hot = np.zeros(5); one_hot[CLASS_MAP[cid]] = 1
        final_f = np.hstack([physics, np.tile(one_hot, (len(physics), 1))]).astype(np.float32)

        for i in range(len(final_f) - TOTAL_REQ + 1):
            X_list.append(final_f[i : i + SEQ_LENGTH])
            y_list.append(final_f[i + SEQ_LENGTH : i + TOTAL_REQ, :2])

    return X_list, y_list

# ==========================================
# 🔄 [핵심] 폴더 그룹 처리 함수 (process_group)
# ==========================================
def process_group(folder_paths, model):
    """연속된 폴더 리스트를 하나의 영상처럼 처리"""
    all_img_files = []
    for f_path in folder_paths:
        # 각 폴더 내의 이미지를 찾아서 정렬 후 추가
        imgs = sorted(glob.glob(os.path.join(f_path, "*.jpg")))
        all_img_files.extend(imgs)

    if not all_img_files: return None, None

    raw_list = []
    # ByteTrack을 사용하여 전체 이미지 시퀀스를 한 번에 추적
    for idx, img_path in enumerate(all_img_files):
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONF_THRESHOLD, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            tids = results[0].boxes.id.int().cpu().tolist()
            cids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, tid, cid in zip(boxes, tids, cids):
                if cid in CLASS_MAP:
                    raw_list.append([idx * DT, tid, cid, box[0], box[1], box[2], box[3]])

    return extract_physics_features(raw_list)

# ==========================================
# 🚀 [메인] 실행 로직
# ==========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 A-PAS 폴더 스티칭 추출 시작 (Device: {device})")
    model = YOLO(MODEL_NAME).to(device)

    for split in ["Training", "Validation"]:
        split_path = os.path.join(INPUT_BASE, split)
        if not os.path.exists(split_path): continue
        
        folders = sorted([f for f in os.scandir(split_path) if f.is_dir()], key=lambda x: x.name)
        print(f"\n📂 [{split}] 총 {len(folders)}개 폴더 분석 및 그룹화 중...")

        split_X, split_y = [], []
        continuous_group = []
        last_num = -1

        for folder in folders:
            curr_num = get_folder_num(folder.name)
            
            # 번호가 끊기면 지금까지 모은 그룹 처리
            if last_num != -1 and curr_num != last_num + 1:
                if len(continuous_group) >= 3: # 최소 3초 데이터 확보 시에만
                    print(f"   📦 그룹 처리: {os.path.basename(continuous_group[0])} ~ {os.path.basename(continuous_group[-1])}")
                    fx, fy = process_group(continuous_group, model)
                    if fx:
                        split_X.extend(fx)
                        split_y.extend(fy)
                continuous_group = []
            
            continuous_group.append(folder.path)
            last_num = curr_num
            
        # 마지막 남은 그룹 처리
        if len(continuous_group) >= 3:
            fx, fy = process_group(continuous_group, model)
            if fx:
                split_X.extend(fx)
                split_y.extend(fy)

        # 결과 저장
        if split_X:
            np.save(f"X_{split}_17f.npy", np.array(split_X, dtype=np.float32))
            np.save(f"y_{split}_17f.npy", np.array(split_y, dtype=np.float32))
            print(f"💾 [{split}] 완료! 누적 시퀀스: {len(split_X)}")

if __name__ == "__main__":
    main()