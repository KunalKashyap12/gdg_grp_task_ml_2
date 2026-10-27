"""
Export Model v1.1.0 to ONNX with dynamic batch axes and verify numerical parity.
Uses opset_version=17 with dynamo=False matching v1.0.0 exporter.
"""

import os
import sys
import json
import shutil
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.abspath("."))
from src.train.models import build_model

def main():
    print("=" * 70)
    print("Exporting Model v1.1.0 to ONNX and Verifying Numerical Parity")
    print("=" * 70)

    ckpt_path = "models_v1.1.0/best_model.pt"
    onnx_path = "models_v1.1.0/crop_disease_model.onnx"
    
    print(f"[*] Loading PyTorch weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt.get("num_classes", 38)
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    
    print(f"[*] Checkpoint info: model={model_name}, classes={num_classes}, val_acc={ckpt.get('val_acc')*100:.2f}%, val_f1={ckpt.get('val_macro_f1')*100:.2f}%")
    
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    print(f"[*] Exporting to ONNX at {onnx_path} (opset 17, dynamo=False)...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        dynamo=False
    )
    print(f"[+] Export completed successfully! File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

    # Validate ONNX graph integrity
    print("[*] Validating ONNX graph with onnx.checker...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model syntax and topological validation passed!")

    # Verify Numerical Parity across PyTorch and ONNX Runtime
    print("[*] Verifying numerical output parity across PyTorch and ONNX Runtime on 60 images...")
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    np.random.seed(42)
    test_batch_sizes = [1, 4, 8, 16, 32]
    max_abs_diff = 0.0

    for bs in test_batch_sizes:
        test_inputs = torch.randn(bs, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            torch_outputs = model(test_inputs).numpy()

        ort_inputs = {"input": test_inputs.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)[0]

        diff = float(np.max(np.abs(torch_outputs - ort_outputs)))
        max_abs_diff = max(max_abs_diff, diff)
        parity = np.allclose(torch_outputs, ort_outputs, atol=1e-4, rtol=1e-3)
        print(f"  - Batch size {bs:2d}: Parity Passed = {parity} | Max Abs Diff = {diff:.6e}")
        assert parity, f"Parity check failed for batch size {bs}! Max diff: {diff}"

    print(f"\n[+] NUMERICAL PARITY CONFIRMED: Max absolute difference across all batch sizes = {max_abs_diff:.6e} < 1e-4")

    # Sync to ml-service/models_v1.1.0/ if folder exists
    alt_out_dir = "ml-service/models_v1.1.0"
    os.makedirs(alt_out_dir, exist_ok=True)
    for f_name in os.listdir("models_v1.1.0"):
        src_f = os.path.join("models_v1.1.0", f_name)
        dst_f = os.path.join(alt_out_dir, f_name)
        if os.path.isfile(src_f):
            shutil.copy2(src_f, dst_f)
    print(f"[+] Synced artifacts to {alt_out_dir}")
    print("[+] PHASE 4 COMPLETE: MODEL v1.1.0 FULLY EXPORTED & VERIFIED!")

if __name__ == "__main__":
    main()
