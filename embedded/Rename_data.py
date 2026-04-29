# -*- coding: utf-8 -*-
"""
여러 데이터셋을 하나로 합치고 순서대로 번호 매기기
====================================================
사용법:
  1. 아래 DATASETS 리스트에 폴더 경로 입력
  2. py -3.10 renumber_dataset.py
"""

import os
import shutil

# ==========================================
# 설정
# ==========================================

# 합칠 데이터셋 폴더 목록 (순서대로)
DATASETS = [
    "carla_data_v4",   # 데이터 8
    "carla_data13",   # 데이터 9
]

# 합쳐진 결과 저장 폴더
OUTPUT_DIR = "carla_data_v5"


# ==========================================
# 메인
# ==========================================
def main():
    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    counter    = 1
    total_bg   = 0
    total_norm = 0

    for dataset in DATASETS:
        img_dir = os.path.join(dataset, "images")
        lbl_dir = os.path.join(dataset, "labels")

        if not os.path.exists(img_dir):
            print("[스킵] 폴더 없음: " + img_dir)
            continue

        files = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])

        print("\n[" + dataset + "] " + str(len(files)) + "장 처리 중...")

        for fname in files:
            base     = os.path.splitext(fname)[0]
            img_src  = os.path.join(img_dir, fname)
            lbl_src  = os.path.join(lbl_dir, base + ".txt")

            # 새 파일명 (frame_000001 형식으로 통일)
            new_name = "frame_" + "{:06d}".format(counter)
            img_dst  = os.path.join(out_img_dir, new_name + ".jpg")
            lbl_dst  = os.path.join(out_lbl_dir, new_name + ".txt")

            # 이미지 복사
            shutil.copy2(img_src, img_dst)

            # 라벨 복사 (없으면 빈 파일 생성)
            if os.path.exists(lbl_src):
                shutil.copy2(lbl_src, lbl_dst)
            else:
                open(lbl_dst, "w").close()

            # 통계
            is_bg = os.path.exists(lbl_src) and \
                    os.path.getsize(lbl_src) == 0
            if is_bg or base.startswith("bg"):
                total_bg += 1
            else:
                total_norm += 1

            counter += 1

        print("  완료!")

    total = counter - 1

    print("\n" + "=" * 45)
    print("완료!")
    print("=" * 45)
    print("총 파일 수  : " + str(total) + "장")
    print("일반 프레임 : " + str(total_norm) + "장")
    print("배경 프레임 : " + str(total_bg) + "장")
    print("저장 위치   : " + os.path.abspath(OUTPUT_DIR))
    print("=" * 45)


if __name__ == "__main__":
    main()