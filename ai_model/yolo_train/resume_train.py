from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\Users\USER\Desktop\UIJIN_Workfolder\workplace\A-PAS\ai_model\yolo_train\runs\detect\best_v8_5class\weights\last.pt")
    model.train(resume=True)