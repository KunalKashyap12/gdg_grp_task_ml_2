"""
Optuna Hyperparameter Optimization and Final Model v1.2.0 Retraining on NVIDIA GPU.
- Device: NVIDIA GeForce RTX 3050 A Laptop GPU (CUDA).
- Base Model: models_v1.1.1/best_model.pt (38 classes).
- Optuna Study: 10 trials with TPESampler and MedianPruner.
- Search Space:
    * lr_head: [1e-4, 3e-3] (log)
    * lr_backbone: [1e-6, 1e-4] (log)
    * weight_decay: [1e-5, 1e-2] (log)
    * label_smoothing: [0.02, 0.12]
    * unfreeze_blocks: [1, 2, 3]
    * batch_size: [64, 128]
- Retraining: Full 17,858 dataset for 3 epochs using winning hyperparameters.
- Deliverables: models_v1.2.0/ artifacts, calibration, ONNX parity, and held-out benchmarks.
"""

import os
import sys
import json
import time
import copy
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
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

sys.path.insert(0, ".")
from src.train.models import build_model
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_train_transforms(img_size=224):
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

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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

def create_objective(search_df, val_df, class_weights_t, base_state_dict, device):
    def objective(trial):
        # 1. Sample Hyperparameters
        lr_head = trial.suggest_float("lr_head", 1e-4, 3e-3, log=True)
        lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        label_smoothing = trial.suggest_float("label_smoothing", 0.02, 0.12)
        unfreeze_blocks = trial.suggest_categorical("unfreeze_blocks", [1, 2, 3])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])

        # 2. DataLoaders
        train_loader = DataLoader(
            ManifestDataset(search_df, transform=get_train_transforms(224)),
            batch_size=batch_size, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            ManifestDataset(val_df, transform=get_val_transforms(224)),
            batch_size=batch_size, shuffle=False, pin_memory=True
        )

        # 3. Model Setup from in-memory base_state_dict
        model = build_model("mobilenet_v3_large", num_classes=38, pretrained=False)
        model.load_state_dict(copy.deepcopy(base_state_dict))
        model.to(device)

        # Configure Parameter Groups and Unfreezing Depth
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        backbone_params = []
        if unfreeze_blocks == 1:
            for p in model.features[-1].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        elif unfreeze_blocks == 2:
            for p in model.features[-2:].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        elif unfreeze_blocks == 3:
            for p in model.features[-3:].parameters():
                p.requires_grad = True
                backbone_params.append(p)

        optimizer = optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head}
        ], weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=label_smoothing)

        # 4. Train 2 Epochs per Trial with Optuna Pruning
        epochs = 2
        for ep in range(epochs):
            model.train()
            for images, targets in train_loader:
                images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

            # Report metric to Optuna for automated pruning
            trial.report(val_f1, ep)
            if trial.should_prune():
                print(f"  [Trial {trial.number} Ep {ep+1}] Pruned (val_macro_f1={val_f1*100:.2f}%)", flush=True)
                raise optuna.exceptions.TrialPruned()

        print(f"  [Trial {trial.number} Finished] val_loss: {val_loss:.4f} | val_acc: {val_acc*100:.2f}% | val_macro_f1: {val_f1*100:.2f}%", flush=True)
        return val_f1

    return objective

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"[*] Starting Optuna Hyperparameter Optimization on {device}")
    if torch.cuda.is_available():
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 75)

    # Load Data Manifests
    train_df = pd.read_csv("data/manifests/train_v1.1.1.csv")
    val_df = pd.read_csv("data/manifests/val_v1.1.1.csv")
    print(f"[*] Full Train set: {len(train_df)} samples | Val set: {len(val_df)} samples across 38 classes")

    # Create stratified search dataset (100 samples per class = 3,800 images) for rapid trial exploration
    search_indices = []
    for _, group in train_df.groupby("class_id"):
        sampled = group.sample(min(len(group), 100), random_state=42)
        search_indices.extend(sampled.index.tolist())
    search_df = train_df.loc[search_indices].reset_index(drop=True)
    print(f"[*] Optuna Exploration Subset: {len(search_df)} stratified samples (100 per class)", flush=True)

    # Class Weights
    class_counts = train_df["class_id"].value_counts().to_dict()
    num_classes = 38
    total_samples = len(train_df)
    weights = [total_samples / (num_classes * class_counts.get(i, 1)) for i in range(num_classes)]
    class_weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    # Load base checkpoint into memory once
    ckpt_candidates = [
        "models_v1.1.1/best_model.pt",
        "models/archive/v1.1.1/best_model.pt",
        "models_v1.1.0/best_model.pt"
    ]
    base_ckpt_path = next(p for p in ckpt_candidates if os.path.exists(p))
    print(f"[*] Loading base model state from {base_ckpt_path}")
    base_ckpt = torch.load(base_ckpt_path, map_location="cpu")
    base_state_dict = base_ckpt["state_dict"]

    # Optuna Study Setup
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="crop_disease_v120")

    objective_fn = create_objective(search_df, val_df, class_weights_t, base_state_dict, device)

    n_trials = 10
    print(f"[*] Executing {n_trials} Optimization Trials on GPU...")
    t0 = time.time()
    study.optimize(objective_fn, n_trials=n_trials)
    study_duration = time.time() - t0

    print("\n" + "=" * 75)
    print(f"[+] Optuna Study Completed in {study_duration:.1f}s ({study_duration/60:.1f} min)")
    print(f"[+] Best Trial #{study.best_trial.number} | Best Validation Macro F1: {study.best_value*100:.2f}%")
    print("[+] Discovered Optimal Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")
    print("=" * 75)

    # Save Study Artifacts
    os.makedirs("reports/improvement", exist_ok=True)
    study_summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": study.best_trial.number,
        "best_macro_f1": float(study.best_value),
        "best_params": study.best_params,
        "duration_seconds": float(study_duration),
        "trials": [
            {
                "number": t.number,
                "value": float(t.value) if t.value is not None else None,
                "params": t.params,
                "state": str(t.state)
            }
            for t in study.trials
        ]
    }
    with open("reports/improvement/optuna_study.json", "w") as f:
        json.dump(study_summary, f, indent=2)
    print("[+] Saved reports/improvement/optuna_study.json")

    # Retrain Final Model v1.2.0 on FULL Training Dataset (17,858 images)
    print("\n" + "=" * 75)
    print("[*] Retraining Final Model v1.2.0 on FULL Dataset (17,858 samples) with Optimal Hyperparameters...")
    print("=" * 75)
    out_dir = "models_v1.2.0"
    os.makedirs(out_dir, exist_ok=True)

    best_p = study.best_params
    batch_size = best_p["batch_size"]
    train_loader = DataLoader(
        ManifestDataset(train_df, transform=get_train_transforms(224)),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        ManifestDataset(val_df, transform=get_val_transforms(224)),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )

    model = build_model("mobilenet_v3_large", num_classes=38, pretrained=False)
    model.load_state_dict(copy.deepcopy(base_state_dict))
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    backbone_params = []
    if best_p["unfreeze_blocks"] == 1:
        for p in model.features[-1].parameters():
            p.requires_grad = True
            backbone_params.append(p)
    elif best_p["unfreeze_blocks"] == 2:
        for p in model.features[-2:].parameters():
            p.requires_grad = True
            backbone_params.append(p)
    elif best_p["unfreeze_blocks"] == 3:
        for p in model.features[-3:].parameters():
            p.requires_grad = True
            backbone_params.append(p)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": best_p["lr_backbone"]},
        {"params": model.classifier.parameters(), "lr": best_p["lr_head"]}
    ], weight_decay=best_p["weight_decay"])

    epochs_final = 3
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_final, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=best_p["label_smoothing"])

    for ep in range(1, epochs_final + 1):
        t_ep = time.time()
        model.train()
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        print(f"  [v1.2.0 Epoch {ep}/{epochs_final}] Finished in {time.time()-t_ep:.1f}s | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_f1*100:.2f}%")

    # Save Checkpoint
    best_model_path = os.path.join(out_dir, "best_model.pt")
    torch.save({
        "model_name": "mobilenet_v3_large",
        "model_version": "1.2.0",
        "num_classes": 38,
        "state_dict": model.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "optuna_params": best_p
    }, best_model_path)
    print(f"[+] Saved {best_model_path}")

    # Temperature Scaling on Validation Split Only
    print("\n[*] Fitting Temperature Scaling on Validation Split Only...")
    from src.calibrate.temperature_scaling import ModelWithTemperature
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

    # Parity Check
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
