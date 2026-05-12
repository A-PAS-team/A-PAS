"""
A-PAS: YOLOv8n CARLA Fine-tuning
=================================
Pseudo-labeling으로 생성된 CARLA 데이터셋으로
YOLOv8n을 fine-tuning합니다.

사용법:
  python train_yolo_carla.py

결과:
  runs/detect/apas_carla/weights/best.pt  → fine-tuned 모델
"""

from ultralytics import YOLO
import os
import torch
torch.cuda.set_per_process_memory_fraction(0.5)
import shutil
# ==========================================
# ⚙️ [설정]
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 베이스 모델 (COCO pretrained)
BASE_MODEL = "yolov8n.pt"

# 데이터셋 (prepare_yolo_data.py에서 생성한 것)
DATA_YAML = os.path.join(BASE_DIR, "dataset_v8_5class", "data.yaml")

# 학습 설정
EPOCHS     = 150         # 500~1000장이면 50 에포크면 충분
IMGSZ      = 1280        # Hailo-8 배포 기준 640
BATCH_SIZE = 6         # RTX 5060이면 16~32 가능
PATIENCE   = 30         # Early stopping

# 프로젝트명 (결과 저장 폴더)
PROJECT = os.path.join(BASE_DIR, "runs", "detect")
NAME    = "best_v8_5class"

# ==========================================
# 🚀 [학습]
# ==========================================
def main():
    # 데이터셋 확인
    if not os.path.exists(DATA_YAML):
        print(f"❌ {DATA_YAML} 이 없습니다!")
        print("   먼저 auto_label_carla.py를 실행하세요.")
        return

    print("=" * 50)
    print("🚀 A-PAS YOLOv8n CARLA Fine-tuning")
    print("=" * 50)
    print(f"  Base Model : {BASE_MODEL}")
    print(f"  Dataset    : {DATA_YAML}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Image Size : {IMGSZ}")
    print(f"  Batch Size : {BATCH_SIZE}")
    print("=" * 50)

    # 모델 로드 (COCO pretrained)
    model = YOLO(BASE_MODEL)

    # Fine-tuning 시작
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,

        # === 핵심: Fine-tuning 전략 ===
        # 작은 데이터셋(500~1000장)에서의 오버피팅 방지
        lr0=0.001,          # 초기 LR (pretrained이니까 낮게)
        lrf=0.01,           # 최종 LR 비율
        warmup_epochs=5,    # 워밍업

        # Data augmentation (CARLA→실세계 도메인 갭 줄이기)
        hsv_h=0.015,        # 색상 변환
        hsv_s=0.7,          # 채도 변환
        hsv_v=0.4,          # 밝기 변환
        degrees=5.0,        # 회전 (약간만)
        translate=0.1,      # 이동
        scale=0.3,          # 스케일
        flipud=0.0,         # 상하 반전 (CCTV 앵글이니 비활성)
        fliplr=0.5,         # 좌우 반전
        mosaic=1.0,         # 모자이크 (과하면 안 좋음)
        bgr=0.1,
        erasing=0.2,


        # 저장
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        save=True,
        save_period=10,     # 10 에포크마다 체크포인트

        # 기타
        workers=4,
        verbose=True,
    )

    # 결과 경로
    best_path = os.path.join(PROJECT, NAME, "weights", "best.pt")
    print("\n" + "=" * 50)
    print("🎉 Fine-tuning 완료!")
    print("=" * 50)
    print(f"  Best model : {best_path}")
    print(f"  결과 폴더  : {os.path.join(PROJECT, NAME)}")
    print()
    
    shutil.copy(best_path, os.path.join(PROJECT, NAME, "weights", "best_v8.pt"))
    # === ONNX 변환 (Hailo DFC 입력용) ===
    print("🔄 ONNX 변환 중...")
    best_model = YOLO(best_path)
    onnx_path = best_model.export(
        format="onnx",
        imgsz=1280,
        opset=13,       # Hailo DFC 호환
        simplify=True,
    )
    print(f"  ✅ ONNX 저장: {onnx_path}")

    print("\n📋 다음 단계:")
    print("  1. ONNX → HEF 변환 (Hailo DFC, x86 리눅스 PC 필요)")
    print("     hailo optimize yolov8n_carla.onnx --hw-arch hailo8")
    print("     hailo compiler yolov8n_carla.har")
    print("  2. 또는 Hailo Model Zoo CLI로 변환:")
    print("     hailomz compile yolov8n --ckpt best.pt --hw-arch hailo8")
    print("  3. HEF를 RPi5로 복사 후 main_hailo.py에서 경로 변경")


if __name__ == "__main__":
    main()
