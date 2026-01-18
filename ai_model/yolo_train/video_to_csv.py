import cv2
import torch
import csv
import os
import glob
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정] 경로 설정 (폴더 구조에 맞게 수정)
# ==========================================
# 영상이 들어있는 폴더 (입력)
INPUT_FOLDER = "../../raw_data" 

# CSV가 저장될 폴더 (출력)
OUTPUT_FOLDER = "data" 

# YOLO 모델
MODEL_NAME = "yolov8n.pt"
CONF_THRESHOLD = 0.3  # 인식 정확도 기준
# ==========================================

def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    # CSV 파일 생성
    f = open(output_path, 'w', newline='')
    wr = csv.writer(f)
    # 헤더 작성 (나중에 학습할 때 이 이름들을 씁니다)
    wr.writerow(['frame', 'track_id', 'class_id', 'x_center', 'y_center', 'width', 'height'])

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   ▶ 처리 시작: {os.path.basename(video_path)} ({total_frames} frames)")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        
        # YOLO 추적 (Tracking)
        # persist=True: 이전 프레임의 ID를 기억함
        results = model.track(frame, persist=True, verbose=False, conf=CONF_THRESHOLD)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  # x중심, y중심, 너비, 높이
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            
            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                # 사람(0), 자동차(2), 오토바이(3), 버스(5), 트럭(7)만 저장
                if class_id in [0, 2, 3, 5, 7]:
                    x_center, y_center, w, h = box
                    
                    # CSV에 한 줄 쓰기
                    # 소수점 2자리까지만 저장 (용량 절약)
                    wr.writerow([
                        frame_count, 
                        track_id, 
                        class_id, 
                        round(x_center, 2), 
                        round(y_center, 2), 
                        round(w, 2), 
                        round(h, 2)
                    ])

        # 진행률 표시 (100프레임마다)
        if frame_count % 100 == 0:
            print(f"      진행중... {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)", end='\r')

    cap.release()
    f.close()
    print(f"\n   ✅ 완료! 저장됨: {output_path}")

def main():
    # 1. GPU 가속 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 시스템 가동 (Device: {device})")
    print(f"📂 입력 폴더: {os.path.abspath(INPUT_FOLDER)}")
    print(f"📂 출력 폴더: {os.path.abspath(OUTPUT_FOLDER)}")

    # 2. 출력 폴더 없으면 만들기
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 3. 모델 로딩 (한 번만 로딩해서 계속 씀)
    model = YOLO(MODEL_NAME)
    model.to(device)

    # 4. 폴더 내 모든 영상 파일 찾기 (*.mp4, *.avi 등)
    # 필요하면 확장자 추가하세요
    video_files = glob.glob(os.path.join(INPUT_FOLDER, "*.mp4")) + \
                  glob.glob(os.path.join(INPUT_FOLDER, "*.avi"))

    if not video_files:
        print("⚠️ 처리할 영상이 없습니다! raw_data 폴더를 확인해주세요.")
        return

    print(f"대상 영상 파일: {len(video_files)}개 발견됨.")
    print("="*40)

    # 5. 하나씩 꺼내서 처리 (Loop)
    for i, video_path in enumerate(video_files):
        # 파일명 추출 (예: video_01.mp4)
        filename = os.path.basename(video_path)
        name_only = os.path.splitext(filename)[0]
        
        # 저장할 CSV 이름 만들기 (예: normal_data_video_01.csv)
        # 앞에 'normal_data_'를 붙여서 나중에 학습 코드 패턴에 맞춤
        save_name = f"normal_data_{name_only}.csv"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        
        print(f"[{i+1}/{len(video_files)}] 변환 중: {filename}")
        
        process_video(video_path, save_path, model)
        print("-" * 40)

    print("🎉 모든 변환 작업이 끝났습니다!")

if __name__ == "__main__":
    main()