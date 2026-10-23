# Phase 6 Report: Baseline vs. Transfer Learning Comparison

## 1. Executive Summary & Recommended Model
- **Selected Model**: **`mobilenet_v3_large`**
- **Validation Accuracy**: **89.87%**
- **Validation Macro-F1**: **89.34%**
- **Parameter Count**: **4.25 Million**
- **Model Checkpoint Size**: **16.41 MB**
- **CPU Inference Latency**: **33.07 ms / image**
- **Training Time**: 915.2 s

> [!IMPORTANT]
> **Deployment Rationale**: MobileNetV3-Large clearly outperforms the baseline CNN (+70.9% accuracy gain) and EfficientNet-B0 (+12.4% accuracy gain) while remaining exceptionally lightweight (16.4 MB) with ultra-fast 33ms CPU inference latency. This makes MobileNetV3-Large the optimal candidate for low-resource cloud deployments and edge REST serving.

---

## 2. Benchmark Comparison Table

| Architecture | Validation Accuracy | Validation Macro-F1 | Parameters (M) | Model Size (MB) | CPU Latency (ms) | Training Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`mobilenet_v3_large`** | **89.87%** | **89.34%** | **4.25 M** | **16.41 MB** | **33.07 ms** | **915.2 s** |
| `efficientnet_b0` | 77.51% | 76.14% | 4.05 M | 15.75 MB | 48.07 ms | 584.0 s |
| `baseline_cnn` | 18.96% | 13.99% | 0.43 M | 1.64 MB | 23.97 ms | 1192.9 s |

*Comparison figure saved at:* [`reports/model_comparison.png`](file:///c:/Users/HP/OneDrive/Desktop/gdg-grp-task-222/reports/model_comparison.png)
