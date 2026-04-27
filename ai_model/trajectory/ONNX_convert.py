# ONNX_convert_residual.py
import torch
import torch.nn as nn
import os

# ==========================================
# ⚙️ [설정] tr_residual_30fps.py와 동일
# ==========================================
INPUT_SIZE  = 17
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
SEQ_LENGTH  = 30
PRED_LENGTH = 60

MODEL_PATH = "models/Residual/0427/best_residual_30fps_pred2s.pth"
ONNX_PATH  = "models/Residual/0427/best_residual_30fps_pred2s.onnx"

# ==========================================
# 🧠 [모델 정의] 학습 코드와 완전 동일
# ==========================================
class ResidualLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
            batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LENGTH * 2)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        offset   = self.fc(h_n[-1]).view(-1, PRED_LENGTH, 2)
        last_pos = x[:, -1, :2].unsqueeze(1)  # 마지막 관찰 위치
        return last_pos + offset               # Residual: 위치 + 오프셋

# ==========================================
# 🔄 [변환]
# ==========================================
def convert():
    device = torch.device("cpu")

    print("📦 Residual LSTM 모델 로드 중...")
    model = ResidualLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"  ✅ 로드 완료: {MODEL_PATH}")
    print(f"  📐 파라미터: {sum(p.numel() for p in model.parameters()):,}개")

    # 더미 입력
    dummy_input = torch.randn(1, SEQ_LENGTH, INPUT_SIZE)
    print(f"  📐 입력 shape: {dummy_input.shape}")

    # ONNX 변환
    print("🔄 ONNX 변환 중...")
    os.makedirs(os.path.dirname(ONNX_PATH), exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"  ✅ 변환 완료: {ONNX_PATH}")

    # 검증
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX 검증 통과!")

    # ONNX Runtime 추론 테스트
    import onnxruntime as ort
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    diff = abs(torch_output - onnx_output).max()
    size_mb = os.path.getsize(ONNX_PATH) / (1024 * 1024)

    print(f"\n📊 검증 결과")
    print(f"  입력 : (batch, {SEQ_LENGTH}, {INPUT_SIZE})")
    print(f"  출력 : (batch, {PRED_LENGTH}, 2)")
    print(f"  PyTorch ↔ ONNX 최대 오차: {diff:.8f}")
    print(f"  파일 크기: {size_mb:.2f}MB")
    print(f"\n🎉 완료! {ONNX_PATH} → RPi5로 전송하세요.")

if __name__ == "__main__":
    convert()