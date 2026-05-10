import os
from collections import Counter

DST = r"C:\Users\USER\Desktop\UIJIN_Workfolder\workplace\dataset_v8_5class"

cls_counts = Counter()
for split in ["train", "val"]:
    lbl_dir = os.path.join(DST, "labels", split)
    for f in os.listdir(lbl_dir):
        if f.endswith(".txt"):
            with open(os.path.join(lbl_dir, f)) as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts:
                        cls_counts[int(float(parts[0]))] += 1

names = ['Person', 'Car', 'Motorcycle', 'Large_Veh', 'Crosswalk']
print("📊 클래스별 bbox 분포:")
for i in range(5):
    print(f"  {i}: {names[i]:12s} → {cls_counts.get(i, 0):,}개")