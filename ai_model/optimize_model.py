import torch
import torch.nn as nn

# 1. 기존 모델 클래스 정의 (main.py와 동일해야 함)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 20) # PRED_LENGTH * 2

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 2. 모델 로드
device = torch.device("cpu") # 라파는 CPU 환경
model = LSTMModel(2, 128, 2)
model.load_state_dict(torch.load("models/best_trajectory_model.pth", map_location=device))

# 3. 동적 양자화 적용 (LSTM 레이어 타겟)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)

# 4. 저장
torch.save(quantized_model.state_dict(), "models/best_model_quantized.pth")
print("✅ 경량화 완료! 용량을 확인해 보세요.")