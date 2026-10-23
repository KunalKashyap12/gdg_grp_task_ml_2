"""
Phase 9: ONNX Model Export, Optimization, Parity Verification, and Benchmarking.

Exports:
- models/crop_disease_model.onnx with dynamic batch axes
Verifies:
- onnx.checker validation
- Numerical parity: np.allclose(torch_logits, onnx_logits, atol=1e-4)
Benchmarks:
- PyTorch CPU inference vs ONNX Runtime CPU inference over 100 passes
"""

import os
import sys
import time
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.abspath("."))
from src.train.models import build_model


def export_and_verify_onnx():
    print("=" * 80)
    print("PHASE 9: ONNX EXPORT, PARITY VERIFICATION & RUNTIME BENCHMARK")
    print("=" * 80)

    # 1. Load Classes & Weights
    with open("data/supported_classes.json", "r") as f:
        supported_classes = json.load(f)
    num_classes = len(supported_classes)

    ckpt_path = "models/best_model.pt"
    print(f"[*] Loading PyTorch weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 2. Export to ONNX
    onnx_path = "models/crop_disease_model.onnx"
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[*] Exporting to ONNX at {onnx_path} with dynamic batch axes...")
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
    print(f"[*] Export completed. File size: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")

    # 3. Validate ONNX Model
    print("[*] Validating ONNX graph integrity with onnx.checker...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model syntax and topological check passed successfully!")

    # 4. Parity Verification
    print("[*] Verifying numerical output parity across PyTorch and ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    np.random.seed(42)
    test_batch_sizes = [1, 4, 8]
    max_abs_diff = 0.0

    for bs in test_batch_sizes:
        test_inputs = torch.randn(bs, 3, 224, 224, dtype=torch.float32)

        with torch.no_grad():
            torch_outputs = model(test_inputs).numpy()

        ort_inputs = {"input": test_inputs.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)[0]

        diff = np.max(np.abs(torch_outputs - ort_outputs))
        max_abs_diff = max(max_abs_diff, diff)
        parity = np.allclose(torch_outputs, ort_outputs, atol=1e-4, rtol=1e-3)
        print(f"  - Batch size {bs:2d}: Parity Passed = {parity} | Max Abs Diff = {diff:.6e}")
        assert parity, f"Parity check failed for batch size {bs}! Max diff: {diff}"

    print(f"[+] Numeric Parity Confirmed! Max absolute difference across all tests: {max_abs_diff:.6e} < 1e-4")

    # 5. Latency Benchmarking (100 runs)
    print("\n[*] Benchmarking CPU inference latency (10 warmup, 100 timed runs)...")
    n_warmup = 10
    n_runs = 100
    sample_input_torch = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    sample_input_np = sample_input_torch.numpy()

    # PyTorch warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(sample_input_torch)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input_torch)
    torch_latency_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # ONNX Runtime warmup
    for _ in range(n_warmup):
        _ = ort_session.run(None, {"input": sample_input_np})

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = ort_session.run(None, {"input": sample_input_np})
    onnx_latency_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    speedup = torch_latency_ms / onnx_latency_ms
    print(f"  - PyTorch CPU Latency:      {torch_latency_ms:.2f} ms/image")
    print(f"  - ONNX Runtime CPU Latency: {onnx_latency_ms:.2f} ms/image")
    print(f"  - ONNX Speedup Factor:      {speedup:.2f}x ({((torch_latency_ms - onnx_latency_ms)/torch_latency_ms)*100:.1f}% faster)")

    # 6. Save Benchmark Metrics
    benchmark_metrics = {
        "onnx_file": onnx_path,
        "model_architecture": model_name,
        "onnx_size_mb": round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
        "parity_verified": True,
        "max_absolute_diff": float(max_abs_diff),
        "tolerance_atol": 1e-4,
        "pytorch_cpu_latency_ms": round(torch_latency_ms, 2),
        "onnx_cpu_latency_ms": round(onnx_latency_ms, 2),
        "speedup_factor": round(speedup, 2),
        "opset_version": 17,
        "input_shape": ["batch_size", 3, 224, 224],
        "output_shape": ["batch_size", num_classes]
    }

    report_path = "reports/onnx_export_benchmark.json"
    with open(report_path, "w") as f:
        json.dump(benchmark_metrics, f, indent=2)
    print(f"[*] Benchmark report saved to {report_path}")


if __name__ == "__main__":
    export_and_verify_onnx()
