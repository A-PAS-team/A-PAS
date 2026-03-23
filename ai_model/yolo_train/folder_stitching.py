import os
import glob
import pandas as pd
import numpy as np

def get_folder_num(folder_name):
    """폴더명 끝의 4자리 번호를 추출 (예: C0189 -> 189)"""
    try:
        return int(folder_name.split('_')[-1][1:])
    except:
        return -1

def main_stitching():
    # ... [YOLO 모델 로드 부분 동일] ...
    
    for split in ["Training", "Validation"]:
        split_path = os.path.join(INPUT_BASE, split)
        if not os.path.exists(split_path): continue
        
        # 1. 폴더 목록 가져오기 및 번호순 정렬
        folders = sorted([f for f in os.scandir(split_path) if f.is_dir()], key=lambda x: x.name)
        
        continuous_group = []
        last_num = -1

        for folder in folders:
            curr_num = get_folder_num(folder.name)
            
            # 번호가 이어지지 않으면 지금까지 모은 그룹 처리 후 초기화
            if last_num != -1 and curr_num != last_num + 1:
                if len(continuous_group) >= 3: # 최소 3개 폴더(30프레임) 이상일 때만
                    process_group(continuous_group, model) # 누적 이미지 처리 함수
                continuous_group = []
            
            continuous_group.append(folder.path)
            last_num = curr_num
            
        # 마지막 남은 그룹 처리
        if len(continuous_group) >= 3:
            process_group(continuous_group, model)