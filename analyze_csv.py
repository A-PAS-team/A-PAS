import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("data_result.csv")

# 자동차(class_id=2) 데이터만 뽑기
cars = df[df['class_id'] == 2]

# 트래킹 ID별로 그룹 묶기
for track_id, group in cars.groupby('track_id'):
    # 프레임 차이가 1인 데이터끼리 비교해서 이동 거리(속도) 계산
    # diff() 함수는 앞뒤 행의 차이를 구해줍니다.
    group = group.sort_values('frame')
    group['speed'] = group['y_center'].diff().abs() # 절대값 (위로가나 아래로가나 속도는 양수)
    
    max_speed = group['speed'].max()
    avg_speed = group['speed'].mean()
    
    print(f"🚗 자동차 ID {track_id}번")
    print(f"   - 최고 속도: {max_speed:.5f}")
    print(f"   - 평균 속도: {avg_speed:.5f}")
    print("-" * 20)