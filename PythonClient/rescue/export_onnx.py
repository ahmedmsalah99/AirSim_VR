"""Convert a YOLO11 PyTorch checkpoint (.pt) to ONNX and check the result.

    pip install ultralytics onnx onnxslim onnxruntime
    python export_onnx.py path/to/model.pt

The .onnx file is written next to the .pt. The script then runs the same
input through the PyTorch model and the ONNX model and checks that the
outputs match.

The exported model takes a 1x3xHxW float32 RGB image scaled to [0, 1]
(letterboxed to H=W=imgsz) and returns 1x(4+num_classes)xN raw predictions:
box centre x, centre y, width, height in input pixels, then one score per
class. Non-maximum suppression is not included; the detector does it.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, help="YOLO11 .pt checkpoint")
    parser.add_argument("--imgsz", type=int, default=640, help="square input size the model was trained at")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    print(f"classes: {model.names}")
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True, dynamic=False))
    print(f"wrote {onnx_path}")

    verify(args.weights, onnx_path, args.imgsz)


def verify(weights, onnx_path, imgsz):
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    out = session.get_outputs()[0]
    print(f"ONNX input  {inp.name}: {inp.shape} {inp.type}")
    print(f"ONNX output {out.name}: {out.shape} {out.type}")

    x = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    onnx_out = session.run(None, {inp.name: x})[0]

    # Reload: export() fuses layers of the model it was given.
    torch_model = YOLO(str(weights)).model.float().eval()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(x))
    if isinstance(torch_out, (list, tuple)):
        torch_out = torch_out[0]
    torch_out = torch_out.numpy()

    diff = np.abs(onnx_out - torch_out).max()
    print(f"max |ONNX - PyTorch| = {diff:.2e}")
    if onnx_out.shape != torch_out.shape or diff > 1e-2:
        raise SystemExit("ONNX output does not match PyTorch output")
    print("OK: ONNX output matches PyTorch")


if __name__ == "__main__":
    main()
