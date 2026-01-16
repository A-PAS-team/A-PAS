import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"GPU 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"잡힌 GPU 이름: {torch.cuda.get_device_name(0)}")
    