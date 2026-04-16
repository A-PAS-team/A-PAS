"""
A-PAS: YOLO Fine-tuned 모델 Test 영상 추론 (mp4 저장)
======================================================
conf를 낮춰서 car가 어떻게 탐지되는지 진단하는 용도

사용법:
  python test_yolo_inference.py --video test_video.mp4 --conf 0.05
  python test_yolo_inference.py --video test_video.mp4 --conf 0.3  (기본값)
"""

import cv2
import os
import argparse
from ultralytics import YOLO

# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "apas_carla", "weights", "best.pt"
)

# 클래스 색상 (person=빨강, car=초록, moto=핑크, bus=주황, truck=노랑)
CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 255),
    3: (255, 165, 0),
    4: (255, 255, 0),
}
CLASS_NAMES = {0: "person", 1: "car", 2: "moto", 3: "bus", 4: "truck"}


# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    parser.add_argument("--conf",  type=float, default=0.3, help="신뢰도 임계값")
    parser.add_argument("--output", default=None, help="출력 mp4 경로")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"❌ 영상 없음: {args.video}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        return

    # 출력 경로
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        args.output = os.path.join(
            BASE_DIR, f"{base}_conf{args.conf}_result.mp4"
        )

    print(f"🧠 모델 로드: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 영상 열기
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 영상: {width}x{height} @ {fps:.1f}fps, {total}프레임")
    print(f"🎯 conf: {args.conf}")
    print(f"💾 출력: {args.output}")

    # mp4 writer (H.264)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 통계
    class_counts = {i: 0 for i in range(5)}
    confidence_sum = {i: 0.0 for i in range(5)}
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # YOLO 추론
            results = model(frame, conf=args.conf, verbose=False)

            # 탐지 결과 그리기
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf   = float(boxes.conf[i].item())
                    xyxy   = boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    name  = CLASS_NAMES.get(cls_id, f"cls{cls_id}")

                    # bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 라벨 + confidence
                    label = f"{name} {conf:.2f}"
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        frame, (x1, y1 - text_size[1] - 8),
                        (x1 + text_size[0] + 4, y1), color, -1)
                    cv2.putText(
                        frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # 통계
                    class_counts[cls_id] += 1
                    confidence_sum[cls_id] += conf

            # conf 정보 표시
            cv2.putText(frame, f"conf={args.conf}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (255, 255, 255), 2)

            writer.write(frame)

            if frame_count % 30 == 0 or frame_count == total:
                pct = frame_count / total * 100
                print(f"  [{frame_count}/{total}] {pct:.1f}%", end='\r')

    finally:
        cap.release()
        writer.release()

    # 통계 출력
    print("\n\n" + "=" * 50)
    print(f"📊 탐지 통계 (conf={args.conf})")
    print("=" * 50)
    print(f"{'클래스':<12} {'탐지수':>10} {'평균conf':>10}")
    print("-" * 50)
    total_detections = 0
    for cls_id in range(5):
        count = class_counts[cls_id]
        total_detections += count
        avg_conf = confidence_sum[cls_id] / count if count > 0 else 0
        print(f"{CLASS_NAMES[cls_id]:<12} {count:>10,} {avg_conf:>10.3f}")
    print("-" * 50)
    print(f"{'TOTAL':<12} {total_detections:>10,}")
    print(f"\n💾 저장됨: {args.output}")


if __name__ == "__main__":
    main()
