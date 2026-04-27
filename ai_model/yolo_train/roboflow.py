from roboflow import Roboflow 
rf = Roboflow(api_key="zObw3qyq4sdlU1oxqA4U")
project = rf.workspace().project("My First Project")
project.upload("ai_model/yolo_train/carla_dataset_v5/train/images", split="train")
project.upload("ai_model/yolo_train/carla_dataset_v5/val/images", split="valid")