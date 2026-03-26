# convert_to_onnx.py
import torch
import torch.nn as nn

# ==========================================
# ⚙️ [설정] tr_trajectory.py와 동일해야 함
# ==========================================
INPUT_SIZE  = 17
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
SEQ_LENGTH  = 60
PRED_LENGTH = 30

MODEL_PATH  = "models/best_trajectory_model.pth"
ONNX_PATH   = "models/best_trajectory_model.onnx"

# ==========================================
# 🧠 [모델 정의] tr_trajectory.py와 완전 동일
# ==========================================
class TrajectoryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)

# ==========================================
# 🔄 [변환]
# ==========================================
def convert():
    device = torch.device("cpu")  # ONNX 변환은 CPU로

    # 1. 모델 로드
    print("📦 모델 로드 중...")
    model = TrajectoryLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("  ✅ 모델 로드 완료")

    # 2. 더미 입력 생성 (shape 확인용)
    dummy_input = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)  # (batch=1, 20, 17)
    print(f"  📐 입력 shape: {dummy_input.shape}")

    # 3. ONNX 변환
    print("🔄 ONNX 변환 중...")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,        # ← 12에서 18로 변경
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,            # ← 추가 (dynamic_axes 경고 제거)
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"  ✅ 변환 완료: {ONNX_PATH}")

    # 4. 검증
    import onnx
    model_onnx = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model_onnx)
    print("  ✅ ONNX 모델 검증 통과!")

    # 5. 입출력 shape 확인
    print("\n📊 모델 입출력 정보")
    print(f"  입력: (batch, {SEQ_LENGTH}, {INPUT_SIZE})")
    print(f"  출력: (batch, {PRED_LENGTH}, 2)")
    print(f"\n🎉 완료! {ONNX_PATH} 파일을 확인하세요.")

if __name__ == "__main__":
    convert()