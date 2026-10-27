"""
Retrain MobileNetV3-Large as Model v1.1.1 with Orange added to Unsupported_crop class.
- Training split: data/manifests/train_v1.1.1.csv (17,858 samples)
- Validation split: data/manifests/val_v1.1.1.csv (3,528 samples)
- Output folder: models_v1.1.1/
- Platt calibration and thresholding fitted on validation split only
- Exports models_v1.1.1/crop_disease_model.onnx and verifies numerical parity
"""

import os
import sys
import json
import time
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, ".")
from src.train.models import build_model
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_v111_train_transforms(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=25, p=0.5),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.10, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_v111_val_transforms(img_size=224):
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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    val_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, acc, macro_f1

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[*] Starting Retraining for Model v1.1.1 on {device}")
    print("=" * 70)

    out_dir = "models_v1.1.1"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Manifests
    train_df = pd.read_csv("data/manifests/train_v1.1.1.csv")
    val_df = pd.read_csv("data/manifests/val_v1.1.1.csv")
    print(f"[*] Train set: {len(train_df)} samples | Val set: {len(val_df)} samples across {train_df['class_id'].nunique()} classes")

    torch.set_num_threads(12)
    batch_size = 128
    train_tx = get_v111_train_transforms(224)
    val_tx = get_v111_val_transforms(224)

    train_loader = DataLoader(ManifestDataset(train_df, transform=train_tx), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(ManifestDataset(val_df, transform=val_tx), batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Build MobileNetV3-Large with 38 outputs
    num_classes = 38
    model = build_model("mobilenet_v3_large", num_classes=num_classes, pretrained=False)

    # 3. Transfer weights from v1.1.0
    v110_ckpt_path = "models_v1.1.0/best_model.pt"
    print(f"[*] Loading pretrained v1.1.0 checkpoint: {v110_ckpt_path}")
    ckpt = torch.load(v110_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    # Compute class weights
    class_counts = train_df["class_id"].value_counts().to_dict()
    total_samples = len(train_df)
    weights = [total_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)

    # Check baseline val before fine-tuning
    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"[*] Initial v1.1.0 zero-shot on v1.1.1 Val Set: Loss = {val_loss:.4f}, Acc = {val_acc*100:.2f}%, F1 = {val_f1*100:.2f}%")

    # 4. Two-Stage Fine-Tuning
    # Stage A: Train classifier only (fast adaptation on reject class)
    print("\n[*] Stage A: Fine-tuning classifier head on v1.1.1 dataset...")
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_head = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    
    # Run 1 epoch of Stage A
    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer_head.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_head.step()
        if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
            print(f"  [Stage A Step {step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"  [*] Stage A Val: Loss = {val_loss:.4f}, Acc = {val_acc*100:.2f}%, Macro F1 = {val_f1*100:.2f}%")

    # Stage B: Fine-tune top feature block + classifier with low learning rate
    print("\n[*] Stage B: Fine-tuning top feature blocks + classifier head...")
    # Unfreeze features[-2:]
    for param in model.features[-2:].parameters():
        param.requires_grad = True

    optimizer_full = optim.AdamW([
        {"params": model.features[-2:].parameters(), "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 3e-4}
    ], weight_decay=1e-4)

    model.train()
    for step, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer_full.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer_full.step()
        if (step + 1) % 20 == 0 or (step + 1) == len(train_loader):
            print(f"  [Stage B Step {step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"  [*] Stage B Val: Loss = {val_loss:.4f}, Acc = {val_acc*100:.2f}%, Macro F1 = {val_f1*100:.2f}%")

    # Save best checkpoint
    best_model_path = os.path.join(out_dir, "best_model.pt")
    torch.save({
        "model_name": "mobilenet_v3_large",
        "model_version": "1.1.1",
        "num_classes": 38,
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1
    }, best_model_path)
    print(f"[+] Saved checkpoint to {best_model_path}")

    # 5. Fit Temperature Scaling on Validation Split Only
    print("\n[*] Step 4.5: Fitting Temperature Scaling on Validation Split Only...")
    from src.calibrate.temperature_scaling import ModelWithTemperature
    scaled_model = ModelWithTemperature(model)
    opt_t, val_logits, val_targets = scaled_model.calibrate(val_loader, device=device)
    print(f"[+] Optimal learned temperature for v1.1.1: T = {opt_t:.4f}")

    calibration_path = os.path.join(out_dir, "calibration.json")
    with open(calibration_path, "w") as f:
        json.dump({
            "model_version": "1.1.1",
            "temperature": round(opt_t, 4),
            "calibrated_classes": 38,
            "fitted_split": "val_v1.1.1 (lab + field + orange mixed)"
        }, f, indent=2)
    print(f"[+] Saved {calibration_path}")

    # 6. Re-select Threshold on Validation Split Only
    print("\n[*] Step 4.6: Evaluating Threshold on Validation Split Only...")
    val_probs = np.exp(val_logits / opt_t) / np.sum(np.exp(val_logits / opt_t), axis=1, keepdims=True)
    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)

    thresholds = [0.50, 0.60, 0.65, 0.70, 0.74, 0.80]
    threshold_metrics = []
    for th in thresholds:
        accepted = (val_confs >= th)
        cov = float(np.mean(accepted))
        sel_acc = float(np.mean(val_preds[accepted] == val_targets[accepted])) if np.sum(accepted) > 0 else 1.0
        print(f"  Threshold {th:4.2f} | Cov: {cov*100:5.1f}% | Sel Acc: {sel_acc*100:5.2f}%")
        threshold_metrics.append({"threshold": th, "coverage": cov, "selective_accuracy": sel_acc})

    selected_th = 0.70
    config_path = os.path.join(out_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": "mobilenet_v3_large",
            "model_version": "1.1.1",
            "input_size": [3, 224, 224],
            "num_classes": 38,
            "supported_classes": 37,
            "internal_reject_classes": 1,
            "temperature": round(opt_t, 4),
            "confidence_threshold": selected_th,
            "margin_threshold": 0.15,
            "entropy_threshold": 1.50,
            "threshold_sweep": threshold_metrics,
            "disclaimer": "AI-assisted indication only, not a guaranteed agricultural diagnosis."
        }, f, indent=2)
    print(f"[+] Saved {config_path}")

    # 7. Export to ONNX
    print("\n[*] Step 4.7: Exporting to ONNX...")
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

    # 8. Numerical Parity Check
    import onnxruntime as ort
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    test_imgs = []
    for i in range(50):
        img_t, _ = val_loader.dataset[i]
        test_imgs.append(img_t)
    batch_t = torch.stack(test_imgs).to(device)
    with torch.no_grad():
        pt_logits = model(batch_t).cpu().numpy()
    onnx_inputs = {ort_session.get_inputs()[0].name: batch_t.cpu().numpy()}
    onnx_logits = ort_session.run(None, onnx_inputs)[0]
    max_abs_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    print(f"[+] Parity Check: Max Absolute Diff = {max_abs_diff:.6e} (< 1e-4)")
    assert max_abs_diff < 1e-4, f"Parity violation: {max_abs_diff}"
    print("[+] PARITY PASSED!")

if __name__ == "__main__":
    main()
