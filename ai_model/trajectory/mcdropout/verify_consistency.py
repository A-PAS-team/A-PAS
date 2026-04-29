# check_onnx_meta.py
import onnx
from pathlib import Path

onnx_path = Path(r"C:\Users\dkdld\A-PAS\models\mcdropout\residual_mcd.onnx")
model = onnx.load(str(onnx_path))

print(f"File: {onnx_path}")
print(f"Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"\nIR version: {model.ir_version}")
print(f"Opset: {model.opset_import[0].version}")
print(f"\nInputs:")
for inp in model.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: {shape}")
print(f"\nOutputs:")
for out in model.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: {shape}")

# Stochastic node 확인
random_nodes = [n for n in model.graph.node if 'Random' in n.op_type]
print(f"\nStochastic nodes: {len(random_nodes)}")
for n in random_nodes:
    print(f"  {n.op_type}: {n.name}")