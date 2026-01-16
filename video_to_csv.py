import cv2
import csv
from ultralytics import YOLO
import torch # GPU 확인용

# ==========================================
# 👇 여기만 수정해서 쓰세요! 👇
# ==========================================
VIDEO_PATH = "my_video.mp4"    # 영상 파일 이름
OUTPUT_CSV = "data_result.csv"  # 저장할 파일 이름
MODEL_NAME = "yolov8n.pt"       # 모델 이름
# ==========================================

def extract_data():
    print(f"🔄 모델 로딩 중: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    # 🔥 [핵심 추가] 모델을 미리 GPU로 옮겨버리기 (속도 향상)
    model.to('cuda')

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상을 찾을 수 없습니다: {VIDEO_PATH}")
        return

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 문구 수정함!
    print(f"🎬 분석 시작! (🚀 GPU 모드: {torch.cuda.get_device_name(0)} 동작 중...)")

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'track_id', 'class_id', 'x_center', 'y_center', 'width', 'height'])

        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break 

            # ⭐ device=0 : GPU 사용 명령
            results = model.track(frame, persist=True, verbose=False, device=0)

            if results[0].boxes.id is not None:
                # GPU에 있는 데이터를 CPU로 가져와야(csv 저장 가능) 하므로 .cpu()는 유지해야 함
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    norm_x = round(x / img_width, 4)
                    norm_y = round(y / img_height, 4)
                    norm_w = round(w / img_width, 4)
                    norm_h = round(h / img_height, 4)

                    writer.writerow([frame_idx, track_id, class_id, norm_x, norm_y, norm_w, norm_h])

            # 30프레임마다 진행 상황 출력
            if frame_idx % 30 == 0:
                print(f"⏳ 진행 중... ({frame_idx}/{total_frames})")
            
            frame_idx += 1

    cap.release()
    print(f"✅ 변환 완료! '{OUTPUT_CSV}' 파일을 확인하세요.")

if __name__ == '__main__':
    extract_data()