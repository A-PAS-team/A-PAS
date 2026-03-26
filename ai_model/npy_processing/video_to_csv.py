import cv2
import torch
import csv
import os
import glob
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정]
# ==========================================
# 절대 경로로 변경 (팀원 환경 대응)
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "..", "..", "raw_data")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "csv")

MODEL_NAME     = "yolov8n.pt"
CONF_THRESHOLD = 0.3
MIN_TRACK_LEN  = 90  # 30FPS 기준 최소 3초

# 클래스 정의 (가독성)
TARGET_CLASSES = [0, 2, 3, 5, 7]  # Person, Car, Motorcycle, Bus, Truck

# ==========================================

def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"   ▶ 처리 시작: {os.path.basename(video_path)}")
    print(f"      ({total_frames}프레임 @ {fps:.1f}FPS)")

    raw_tracks  = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=CONF_THRESHOLD
        )

        if results[0].boxes.id is not None:
            boxes     = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                if class_id in TARGET_CLASSES:
                    x, y, w, h = box
                    if track_id not in raw_tracks:
                        raw_tracks[track_id] = []
                    raw_tracks[track_id].append((
                        frame_count, track_id, class_id,
                        round(float(x), 2), round(float(y), 2),
                        round(float(w), 2), round(float(h), 2)
                    ))

        if frame_count % 100 == 0:
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"      진행중... {frame_count}/{total_frames} "
                    f"({progress:.1f}%)", end='\r')
            else:
                print(f"      진행중... {frame_count} 프레임 처리 중", end='\r')

    cap.release()

    # 최소 트래킹 길이 필터링
    filtered = {
        tid: data for tid, data in raw_tracks.items()
        if len(data) >= MIN_TRACK_LEN
    }
    print(f"\n      전체 트랙: {len(raw_tracks)}개 "
          f"→ 필터링 후: {len(filtered)}개")

    # ✅ 프레임 순서로 정렬 후 저장
    all_rows = []
    for data in filtered.values():
        all_rows.extend(data)
    all_rows.sort(key=lambda x: x[0])  # frame 기준 정렬

    with open(output_path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['frame', 'track_id', 'class_id',
                     'x_center', 'y_center', 'width', 'height'])
        for row in all_rows:
            wr.writerow(row)

    print(f"   ✅ 저장됨: {output_path}")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 시스템 가동 (Device: {device})")
    print(f"📂 입력 폴더: {os.path.abspath(INPUT_FOLDER)}")
    print(f"📂 출력 폴더: {os.path.abspath(OUTPUT_FOLDER)}")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    model = YOLO(MODEL_NAME)
    model.to(device)

    video_files = (glob.glob(os.path.join(INPUT_FOLDER, "*.mp4")) +
                   glob.glob(os.path.join(INPUT_FOLDER, "*.avi")))

    if not video_files:
        print("⚠️ 처리할 영상이 없습니다!")
        print(f"   확인 경로: {os.path.abspath(INPUT_FOLDER)}")
        return

    print(f"\n대상 영상: {len(video_files)}개")
    print("=" * 40)

    success_count = 0
    fail_count = 0

    for i, video_path in enumerate(video_files):
        filename  = os.path.basename(video_path)
        name_only = os.path.splitext(filename)[0]
        save_path = os.path.join(OUTPUT_FOLDER, f"{name_only}.csv")

        print(f"[{i+1}/{len(video_files)}] {filename}")

        try:
            process_video(video_path, save_path, model)
            success_count += 1  # ✅
        except Exception as e:
            print(f"⚠️ {filename} 처리 중 에러 발생: {e}")
            print("   다음 영상으로 넘어갑니다...")
            fail_count += 1     # ✅
            continue

        print("-" * 40)

    print("🎉 모든 변환 완료!")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"📂 결과물 위치: {os.path.abspath(OUTPUT_FOLDER)}")

if __name__ == "__main__":
    main()