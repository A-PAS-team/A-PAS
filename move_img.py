import os
import shutil
import glob

# ==========================================
# ⚙️ [설정] 데이터셋 루트 경로
# ==========================================
# 실제 경로를 입력하세요. 윈도우 경로이므로 r을 붙여주는 것이 안전합니다.
root_dir = r"E:\A-pas\178.지능형 인프라 센서 기반 동적객체 인지 데이터\01-1.정식개방데이터\Validation\01.원천데이터\VS"
# ==========================================

def move_jpg_files():
    # 1. 루트 폴더 내의 L_2210_Suwon_B_A_C* 패턴의 폴더들을 모두 찾습니다.
    target_folders = glob.glob(os.path.join(root_dir, "L_2211_Suwon_B_F_C*"))
    
    if not target_folders:
        print("❌ 대상 폴더를 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    print(f"📂 총 {len(target_folders)}개의 시퀀스 폴더를 발견했습니다.")

    moved_total = 0

    for sequence_folder in target_folders:
        # 2. 이동할 파일이 있는 깊은 경로 설정
        # sensor_raw_data/camera/camera_0
        source_dir = os.path.join(sequence_folder, "sensor_raw_data", "camera", "camera_0")
        
        # 만약 해당 경로가 존재하지 않으면 스킵
        if not os.path.exists(source_dir):
            print(f"⚠️ 경로 없음 (스킵): {os.path.basename(sequence_folder)}")
            continue

        # 3. 소스 폴더 내의 모든 jpg 파일 찾기
        jpg_files = glob.glob(os.path.join(source_dir, "*.jpg"))
        
        if not jpg_files:
            continue

        # 4. 파일 이동 실행
        for file_path in jpg_files:
            file_name = os.path.basename(file_path)
            # 목적지: L_2210_Suwon_B_A_C* 폴더 바로 아래
            dest_path = os.path.join(sequence_folder, file_name)
            
            try:
                # 파일 이동 (동일 이름 파일이 있을 경우 덮어쓰지 않도록 체크 가능)
                shutil.move(file_path, dest_path)
                moved_total += 1
            except Exception as e:
                print(f"❌ 이동 실패: {file_name} ({e})")

        print(f"✅ 처리 완료: {os.path.basename(sequence_folder)} ({len(jpg_files)}장 이동)")

    print("-" * 50)
    print(f"🎉 작업 완료! 총 {moved_total}개의 파일을 이동시켰습니다.")

if __name__ == "__main__":
    move_jpg_files()