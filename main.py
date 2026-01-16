import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO

# ==========================================
# ⚙️ [최종 설정] 환경에 맞게 수정하세요
# ==========================================
VIDEO_PATH = "my_video.mp4"     # 영상 파일
MODEL_NAME = "yolov8n.pt"       # 모델 파일
SPEED_THRESHOLD = 0.02          # 🚨 0.02보다 빠르면 위험 (분석 결과 반영)
CONF_THRESHOLD = 0.5            # 정확도 0.5 이상만 인정
# ==========================================

roi_points = []

def click_event(event, x, y, flags, param):
    """마우스로 횡단보도 영역(4점) 찍기"""
    global roi_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(roi_points) < 4:
            roi_points.append((x, y))
            print(f"📍 좌표 찍힘: ({x}, {y})")

def get_speed(track_id, current_y, prev_data):
    """속도 계산 (안전장치 포함)"""
    if track_id in prev_data:
        speed = current_y - prev_data[track_id]
        # ⚠️ 안전장치: nan이거나 비정상적으로 큰 값은 0 처리
        if math.isnan(speed) or abs(speed) > 1.0:
            return 0.0
        return speed
    return 0.0

def is_inside_roi(box, roi_poly):
    """발바닥이 ROI 안에 있는지 확인"""
    x, y, w, h = box
    foot_x, foot_y = int(x), int(y + h/2)
    return cv2.pointPolygonTest(roi_poly, (foot_x, foot_y), False) >= 0

def main():
    global roi_points

    # 1. GPU 가속 활성화
    print(f"🚀 시스템 초기화... (GPU: {torch.cuda.get_device_name(0)})")
    model = YOLO(MODEL_NAME)
    model.to('cuda') 

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 영상을 열 수 없습니다: {VIDEO_PATH}")
        return

    # 2. ROI 설정 (첫 화면)
    ret, first_frame = cap.read()
    if ret:
        print("\n🖱️ 화면에 횡단보도 네 모서리를 클릭하세요! (총 4번)")
        cv2.imshow("Set Crosswalk ROI", first_frame)
        cv2.setMouseCallback("Set Crosswalk ROI", click_event)
        
        while len(roi_points) < 4:
            if cv2.waitKey(10) == 27: return 
            for pt in roi_points:
                cv2.circle(first_frame, pt, 5, (0, 0, 255), -1)
            cv2.imshow("Set Crosswalk ROI", first_frame)
        cv2.destroyWindow("Set Crosswalk ROI")
    
    roi_poly = np.array(roi_points, np.int32)
    prev_y = {}
    frame_h, frame_w = first_frame.shape[:2]

    # ==========================================
    # 🎬 실시간 감지 시작
    # ==========================================
    while True:
        ret, frame = cap.read()
        if not ret: break

        # GPU 추적
        results = model.track(frame, persist=True, verbose=False, device=0, conf=CONF_THRESHOLD)
        annotated_frame = results[0].plot()
        cv2.polylines(annotated_frame, [roi_poly], True, (0, 255, 0), 2)

        pedestrian_detected = False
        danger_detected = False
        warning_message = "SAFE"

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                norm_y = box[1] / frame_h
                speed = get_speed(track_id, norm_y, prev_y) # 모든 객체 속도 계산

                # 🟢 사람(0)인 경우
                if class_id == 0:
                    # [수정됨] 사람이지만 속도가 엄청 빠르다? -> 킥보드/뛰는 사람 -> 위험!
                    if speed > SPEED_THRESHOLD:
                        danger_detected = True
                        warning_message = "FAST OBJECT (Kickboard?)"
                        x1, y1 = int(box[0]), int(box[1] - box[3]/2)
                        cv2.putText(annotated_frame, "FAST!", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 속도는 느린데 횡단보도 안에 있다 -> 보행자
                    elif is_inside_roi(box, roi_poly):
                        pedestrian_detected = True

                # 🚗 차(2,5,7)인 경우
                elif class_id in [2, 5, 7]:
                    if speed > SPEED_THRESHOLD:
                        danger_detected = True
                        warning_message = "FAST CAR DETECTED"
                        x1, y1 = int(box[0]), int(box[1] - box[3]/2)
                        cv2.putText(annotated_frame, "DANGER!", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                prev_y[track_id] = norm_y

        # 3. 상황판 출력
        if danger_detected:
            # 빨간 배경 + 경고
            cv2.rectangle(annotated_frame, (0, 0), (frame_w, 60), (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"WARNING: {warning_message}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        elif pedestrian_detected:
            # 노란 배경 + 보행자 주의
            cv2.rectangle(annotated_frame, (0, 0), (400, 60), (0, 255, 255), -1)
            cv2.putText(annotated_frame, "Pedestrian in Zone", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        else:
            # 초록 글씨 + 안전
            cv2.rectangle(annotated_frame, (0, 0), (200, 60), (0, 0, 0), -1)
            cv2.putText(annotated_frame, "SAFE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("A-PAS Final (RTX 5060)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()