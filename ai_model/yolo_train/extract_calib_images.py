"""
A-PAS 캘리브레이션 이미지 추출 스크립트
- 프로젝트 폴더 내 mp4/avi 영상에서 프레임 추출
- 30프레임 간격으로 샘플링
- 1280x1280으로 리사이즈 (Hailo DFC 양자화용)
"""

import cv2
import os
import glob

# ==========================================
# ⚙️ [설정]
# ==========================================
VIDEO_DIR = "./test_video"                          # 프로젝트 최상위 폴더
OUTPUT_DIR = "calibration_images"        # 출력 폴더
TARGET_SIZE = (1280, 1280)               # Hailo HEF 입력 해상도
FRAME_INTERVAL = 30                      # 30프레임마다 1장 추출
MAX_PER_VIDEO = 80                       # 영상당 최대 추출 수 (총량 조절용)

# ==========================================
# 🚀 [메인]
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 영상 파일 탐색
    video_files = []
    for ext in ["*.mp4", "*.avi", "*.MP4", "*.AVI"]:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, ext)))

    if not video_files:
        print("❌ 영상 파일이 없습니다!")
        print(f"   확인 경로: {os.path.abspath(VIDEO_DIR)}")
        return

    print(f"🎬 발견된 영상: {len(video_files)}개")
    print(f"📂 출력 폴더: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📐 리사이즈: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print(f"⏱️  추출 간격: {FRAME_INTERVAL}프레임마다 1장")
    print("=" * 50)

    total_saved = 0

    for vid_idx, video_path in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"⚠️  열기 실패: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"\n[{vid_idx+1}/{len(video_files)}] {video_name}")
        print(f"   총 {total_frames}프레임 @ {fps:.1f}FPS")

        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % FRAME_INTERVAL == 0:
                # letterbox 리사이즈 (비율 유지 + 패딩)
                resized = letterbox_resize(frame, TARGET_SIZE)

                filename = f"{video_name}_{frame_count:06d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(save_path, resized)

                saved_count += 1
                total_saved += 1

                if saved_count >= MAX_PER_VIDEO:
                    print(f"   ⚠️  최대 {MAX_PER_VIDEO}장 도달, 다음 영상으로")
                    break

        cap.release()
        print(f"   ✅ {saved_count}장 추출")

    print("\n" + "=" * 50)
    print(f"🎉 완료! 총 {total_saved}장 저장됨")
    print(f"📂 경로: {os.path.abspath(OUTPUT_DIR)}")


def letterbox_resize(img, target_size):
    """
    YOLOv8 방식 letterbox 리사이즈
    - 비율 유지하면서 target_size에 맞춤
    - 남는 영역은 (114, 114, 114) 패딩
    """
    h, w = img.shape[:2]
    tw, th = target_size

    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 패딩 (중앙 배치)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    dx = (tw - new_w) // 2
    dy = (th - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    return canvas


if __name__ == "__main__":
    import numpy as np
    main()
