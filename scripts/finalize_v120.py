"""
Finalize Model v1.2.0:
1. Load models_v1.2.0/best_model.pt
2. Run Temperature Scaling on Validation Split (val_v1.1.1.csv)
3. Save calibration.json and model_config.json
4. Export to ONNX (models_v1.2.0/crop_disease_model.onnx)
5. Verify numerical parity (< 1e-4) between PyTorch and ONNX Runtime
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, ".")
from src.train.models import build_model
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD
from src.calibrate.temperature_scaling import ModelWithTemperature

def get_val_transforms(img_size=224):
    return A.Compose([
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

class ManifestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.class_ids = self.df["class_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = resolve_image_path(self.filepaths[idx])
        target = self.class_ids[idx]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img_np = np.array(img.convert("RGB"))
        if self.transform is not None:
            tensor = self.transform(image=img_np)["image"]
        else:
            tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return tensor, target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Finalizing Model v1.2.0 on {device}...")
    
    out_dir = "models_v1.2.0"
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    model = build_model("mobilenet_v3_large", num_classes=38, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    
    # Load optuna study params
    with open("reports/improvement/optuna_study.json", "r") as f:
        study_data = json.load(f)
    best_p = study_data["best_params"]
    print(f"[*] Loaded best params from study: {best_p}")
    
    # Validation DataLoader
    val_df = pd.read_csv("data/manifests/val_v1.1.1.csv")
    val_loader = DataLoader(
        ManifestDataset(val_df, transform=get_val_transforms(224)),
        batch_size=best_p["batch_size"], shuffle=False, pin_memory=True
    )
    
    # Temperature Scaling on Validation Split Only
    print("\n[*] Fitting Temperature Scaling on Validation Split Only...")
    scaled_model = ModelWithTemperature(model)
    opt_t, val_logits, val_targets = scaled_model.calibrate(val_loader, device=device)
    print(f"[+] Optimal learned temperature for v1.2.0: T = {opt_t:.4f}")

    calibration_path = os.path.join(out_dir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump({
            "model_version": "1.2.0",
            "temperature": round(opt_t, 4),
            "calibrated_classes": 38,
            "fitted_split": "val_v1.1.1 (lab + field + orange mixed)"
        }, f, indent=2)
    print(f"[+] Saved {calibration_path}")

    # Threshold Sweep on Validation Split
    val_probs = np.exp(val_logits / opt_t) / np.sum(np.exp(val_logits / opt_t), axis=1, keepdims=True)
    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)
    thresholds = [0.50, 0.60, 0.65, 0.70, 0.74, 0.80]
    threshold_metrics = []
    for th in thresholds:
        accepted = (val_confs >= th)
        cov = float(np.mean(accepted))
        sel_acc = float(np.mean(val_preds[accepted] == val_targets[accepted])) if np.sum(accepted) > 0 else 1.0
        threshold_metrics.append({"threshold": th, "coverage": cov, "selective_accuracy": sel_acc})

    config_path = os.path.join(out_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": "mobilenet_v3_large",
            "model_version": "1.2.0",
            "input_size": [3, 224, 224],
            "num_classes": 38,
            "supported_classes": 37,
            "internal_reject_classes": 1,
            "temperature": round(opt_t, 4),
            "confidence_threshold": 0.70,
            "margin_threshold": 0.15,
            "entropy_threshold": 1.50,
            "threshold_sweep": threshold_metrics,
            "optuna_best_params": best_p,
            "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis."
        }, f, indent=2)
    print(f"[+] Saved {config_path}")

    # Export to ONNX
    print("\n[*] Exporting to ONNX...")
    onnx_path = os.path.join(out_dir, "crop_disease_model.onnx")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False
    )
    print(f"[+] Exported {onnx_path} ({os.path.getsize(onnx_path) / (1024*1024):.2f} MB)")

    # Parity Check on CPU
    print("\n[*] Running ONNX parity verification (PyTorch CPU vs ONNX CPU)...")
    model_cpu = model.to("cpu")
    model_cpu.eval()
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_imgs = []
    for i in range(50):
        img_t, _ = val_loader.dataset[i]
        test_imgs.append(img_t)
    batch_t = torch.stack(test_imgs).to("cpu")
    with torch.no_grad():
        pt_logits = model_cpu(batch_t).numpy()
    onnx_inputs = {ort_session.get_inputs()[0].name: batch_t.numpy()}
    onnx_logits = ort_session.run(None, onnx_inputs)[0]
    max_abs_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    print(f"[+] Parity Check: Max Absolute Diff = {max_abs_diff:.6e} (Threshold: 1e-4)")
    assert max_abs_diff < 1e-4, f"Parity violation: {max_abs_diff}"
    print("[+] ONNX PARITY VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
