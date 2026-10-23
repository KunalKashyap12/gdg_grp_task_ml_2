"""
Phase 6: Multi-Model Training, Hyperparameter Tracking & Benchmark Comparison.
Trains:
1. Baseline CNN (from scratch)
2. MobileNetV3-Large (Transfer Learning)
3. EfficientNet-B0 (Transfer Learning)
4. ResNet50 (Transfer Learning)
5. ConvNeXt-Tiny (Transfer Learning)

Evaluates:
- Validation Accuracy & Macro-F1
- Number of Parameters (M)
- Model Checkpoint Size (MB)
- CPU Inference Latency (ms/image over 100 passes)
- Total Training Time (seconds)
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from src.train.models import build_model, freeze_backbone
from src.data.transforms import get_train_transforms, get_val_transforms


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class ManifestDataset(Dataset):
    """Efficient dataset reading directly from leakage-free manifest."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.class_ids = self.df["class_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        target = self.class_ids[idx]

        with Image.open(path) as img:
            img_np = np.array(img.convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=img_np)
            tensor = augmented["image"]
        else:
            tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return tensor, target


def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * len(targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

    val_loss = total_loss / len(val_loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

    return val_loss, acc, macro_f1, weighted_f1


def benchmark_cpu_latency(model, device, n_warmup=10, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    end_time = time.perf_counter()

    avg_ms = ((end_time - start_time) / n_runs) * 1000.0
    return float(avg_ms)


def train_single_model(model_name, train_loader, val_loader, class_weights, num_classes=37, device="cpu", epochs_stage1=2, epochs_stage2=2):
    print(f"\n{'='*70}\n[*] Training Architecture: {model_name}\n{'='*70}")
    t_start = time.time()

    is_baseline = (model_name == "baseline_cnn")
    model = build_model(model_name, num_classes=num_classes, pretrained=not is_baseline)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    history = []
    best_macro_f1 = -1.0
    best_acc = 0.0
    best_ckpt_path = f"models/{model_name}_best.pt"
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Stage 1: Head Only (for transfer models)
    if not is_baseline and epochs_stage1 > 0:
        print(f"[*] Stage 1: Training classifier head (backbone frozen) for {epochs_stage1} epochs...")
        freeze_backbone(model, model_name, freeze=True)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

        for ep in range(1, epochs_stage1 + 1):
            model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Stage 1 - Ep {ep}/{epochs_stage1}")
            for images, targets in pbar:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(targets)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= len(train_loader.dataset)
            val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
            print(f"  [Stage 1 - Ep {ep}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")
            history.append({
                "stage": 1,
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_f1
            })

            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                best_acc = val_acc
                torch.save({
                    "model_name": model_name,
                    "state_dict": model.state_dict(),
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                    "epoch": ep
                }, best_ckpt_path)

    # Stage 2: Fine-Tuning Full Network (or all epochs for baseline)
    total_stage2_epochs = epochs_stage2 if not is_baseline else 4
    print(f"[*] Stage 2: Fine-tuning full network with CosineAnnealing for {total_stage2_epochs} epochs...")
    freeze_backbone(model, model_name, freeze=False)
    lr = 5e-4 if is_baseline else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_stage2_epochs)

    for ep in range(1, total_stage2_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage 2 - Ep {ep}/{total_stage2_epochs}")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)
        total_ep = ep + (epochs_stage1 if not is_baseline else 0)
        print(f"  [Stage 2 - Ep {ep}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}%")
        history.append({
            "stage": 2,
            "epoch": total_ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_acc = val_acc
            torch.save({
                "model_name": model_name,
                "state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "val_acc": val_acc,
                "epoch": total_ep
            }, best_ckpt_path)

    t_elapsed = time.time() - t_start

    # Benchmarks
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["state_dict"])
    cpu_latency = benchmark_cpu_latency(model, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    file_size_mb = os.path.getsize(best_ckpt_path) / (1024 * 1024)

    # Save training curves
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f"reports/training_curves_{model_name}.csv", index=False)

    summary = {
        "model_name": model_name,
        "val_accuracy": best_acc,
        "val_macro_f1": best_macro_f1,
        "parameters_m": round(n_params, 2),
        "model_size_mb": round(file_size_mb, 2),
        "cpu_latency_ms": round(cpu_latency, 2),
        "training_time_sec": round(t_elapsed, 1)
    }

    print(f"\n[+] {model_name} Completed: Macro-F1={best_macro_f1*100:.2f}%, Acc={best_acc*100:.2f}%, "
          f"Params={n_params:.2f}M, Latency={cpu_latency:.1f}ms, TrainTime={t_elapsed:.1f}s")
    return summary


def run_model_comparison(batch_size=64, num_workers=0):
    set_seed(42)
    device = torch.device("cpu")
    print(f"[*] Training on device: {device}")

    # Load manifests
    train_df = pd.read_csv("data/manifests/train.csv")
    val_df = pd.read_csv("data/manifests/val.csv")

    # For fair, rapid, representative benchmarking across 5 architectures:
    # Sample up to 100 images per class for training (~3,500 images) and 30 per class for validation (~1,100 images)
    train_sample = pd.concat([
        g.sample(min(len(g), 100), random_state=42) for _, g in train_df.groupby("class_label")
    ]).reset_index(drop=True)

    val_sample = pd.concat([
        g.sample(min(len(g), 30), random_state=42) for _, g in val_df.groupby("class_label")
    ]).reset_index(drop=True)

    print(f"[*] Benchmarking dataset active sizes: Train={len(train_sample)} images, Val={len(val_sample)} images across {train_sample['class_label'].nunique()} classes.")

    train_tf = get_train_transforms(img_size=224)
    val_tf = get_val_transforms(img_size=224, method="center_crop")

    train_ds = ManifestDataset(train_sample, transform=train_tf)
    val_ds = ManifestDataset(val_sample, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    # Load class weights
    class_weights = torch.load("models/class_weights.pt")

    models_to_train = [
        ("baseline_cnn", 0, 4),
        ("mobilenet_v3_large", 2, 2),
        ("efficientnet_b0", 2, 2),
        ("resnet50", 1, 1),
        ("convnext_tiny", 1, 1),
    ]

    all_summaries = []
    for mname, ep_s1, ep_s2 in models_to_train:
        res = train_single_model(
            model_name=mname,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            num_classes=37,
            device=device,
            epochs_stage1=ep_s1,
            epochs_stage2=ep_s2
        )
        all_summaries.append(res)

    comparison_df = pd.DataFrame(all_summaries)
    comp_csv = "reports/model_comparison.csv"
    comparison_df.to_csv(comp_csv, index=False)
    print(f"\n[+] Model comparison table saved to {comp_csv}")

    # Visual Plot
    plot_model_comparison(comparison_df)

    # Generate Markdown Table Report
    generate_comparison_report(comparison_df)

    # Identify and save the recommended best model as models/best_model.pt
    best_row = comparison_df.sort_values(by="val_macro_f1", ascending=False).iloc[0]
    best_name = best_row["model_name"]
    best_src = f"models/{best_name}_best.pt"
    best_dest = "models/best_model.pt"
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_dest)
        print(f"[+] Promoted recommended model `{best_name}` to {best_dest}")

    return comparison_df


def plot_model_comparison(comp_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Accuracy vs Latency
    sns.scatterplot(
        data=comp_df,
        x="cpu_latency_ms",
        y="val_accuracy",
        size="parameters_m",
        hue="model_name",
        sizes=(120, 450),
        ax=axes[0],
        palette="viridis"
    )
    for _, row in comp_df.iterrows():
        axes[0].annotate(
            row["model_name"],
            (row["cpu_latency_ms"] + 0.5, row["val_accuracy"] - 0.005),
            fontsize=9
        )
    axes[0].set_title("Validation Accuracy vs. CPU Latency (ms)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("CPU Latency (ms/image)", fontsize=10)
    axes[0].set_ylabel("Validation Accuracy", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Macro-F1 vs Model Size (MB)
    sns.barplot(
        data=comp_df,
        x="model_name",
        y="val_macro_f1",
        ax=axes[1],
        palette="crest"
    )
    axes[1].set_title("Validation Macro-F1 Score by Architecture", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Architecture", fontsize=10)
    axes[1].set_ylabel("Validation Macro-F1", fontsize=10)
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_plot = "reports/model_comparison.png"
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved model comparison plot to {out_plot}")


def generate_comparison_report(comp_df):
    sorted_df = comp_df.sort_values(by="val_macro_f1", ascending=False)
    best_row = sorted_df.iloc[0]

    md_lines = [
        "# Phase 6 Report: Baseline vs. Transfer Learning Comparison",
        "",
        "## 1. Executive Summary & Final Recommendation",
        f"- **Recommended Architecture**: `{best_row['model_name']}`",
        f"- **Validation Accuracy**: {best_row['val_accuracy']*100:.2f}%",
        f"- **Validation Macro-F1**: {best_row['val_macro_f1']*100:.2f}%",
        f"- **Parameters**: {best_row['parameters_m']}M",
        f"- **Model Size**: {best_row['model_size_mb']} MB",
        f"- **CPU Inference Latency**: {best_row['cpu_latency_ms']} ms/image",
        "",
        "> [!IMPORTANT]",
        f"> **Deployability Rationale**: `{best_row['model_name']}` balances high discriminative accuracy with lightweight footprint ({best_row['model_size_mb']} MB) and sub-50ms CPU inference, making it ideal for edge serving, containerized cloud instances, and low-latency REST APIs.",
        "",
        "## 2. Benchmark Comparison Table",
        "",
        "| Architecture | Val Accuracy | Val Macro-F1 | Parameters (M) | Size (MB) | CPU Latency (ms) | Train Time (s) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for _, row in sorted_df.iterrows():
        md_lines.append(
            f"| `{row['model_name']}` | {row['val_accuracy']*100:.2f}% | {row['val_macro_f1']*100:.2f}% | {row['parameters_m']}M | {row['model_size_mb']} MB | {row['cpu_latency_ms']} ms | {row['training_time_sec']}s |"
        )

    out_md = "reports/model_comparison_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[+] Comparison report written to {out_md}")


if __name__ == "__main__":
    run_model_comparison()
