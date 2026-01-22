import os

BASE_DIR = r"E:\raw_data"

def final_unify_rename():
    sub_dirs = ["Training", "Validation"]
    
    for sub in sub_dirs:
        target_path = os.path.join(BASE_DIR, sub)
        if not os.path.exists(target_path): continue

        print(f"\n📂 [{sub}] 폴더 통합 재정렬 시작...")
        items = os.listdir(target_path)
        
        for old_name in items:
            old_full_path = os.path.join(target_path, old_name)
            if not os.path.isdir(old_full_path): continue

            parts = old_name.split('_')
            new_name = None

            # 1. 아직 이름이 안 바뀐 원본 패턴 (L_2210_Suwon_B_F_C0001)
            if old_name.startswith("L_") and len(parts) >= 6:
                year, city, point, attr, seq_id = parts[1], parts[2], parts[3], parts[4], parts[-1]
                new_name = f"{city}_{point}_{year}_{attr}_{seq_id}"

            # 2. 이미 이름이 바뀌어버린 패턴 (Suwon_B_C0155)
            # 이 경우 '연도'와 '특성' 정보가 이름에서 사라졌으므로, 
            # 중복 방지를 위해 현재 이름 뒤에 '_OLD' 등을 붙이거나 
            # 그대로 두되, 'L_' 폴더들이 이 이름을 침범하지 못하게 보호합니다.
            
            if new_name:
                new_full_path = os.path.join(target_path, new_name)
                if not os.path.exists(new_full_path):
                    os.rename(old_full_path, new_full_path)
                    print(f"   ✅ 변경 완료: {old_name} ➔ {new_name}")
                else:
                    # 만약 이름이 겹친다면 뒤에 언더바를 하나 더 붙여서 강제로 구분
                    new_full_path += "_REV"
                    os.rename(old_full_path, new_full_path)
                    print(f"   ⚠️ 중복 회피(이름 뒤에 _REV 추가): {new_name}_REV")

    print("\n🎉 모든 폴더의 고유 이름 확보가 완료되었습니다!")

if __name__ == "__main__":
    final_unify_rename()