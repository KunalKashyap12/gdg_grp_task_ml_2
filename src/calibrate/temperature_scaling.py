"""
Phase 8: Post-Hoc Model Calibration via Temperature Scaling.

Solves overconfidence in modern deep neural networks by learning a single
positive scalar parameter T on validation logits to minimize Negative Log-Likelihood (NLL).

Generates:
- Optimized Temperature parameter T (saved to models/calibration.json)
- Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
- Reliability diagrams before and after calibration (reports/calibration_curves.png)
- Calibrated inference module for serving
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import LBFGS

sys.path.insert(0, os.path.abspath("."))
from src.train.models import build_model
from src.data.transforms import get_val_transforms, resolve_image_path


class ManifestEvalDataset(Dataset):
    """Dataset for extracting validation and test logits."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.filepaths = self.df["filepath"].tolist()
        self.targets = self.df["class_id"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = resolve_image_path(self.filepaths[idx])
        target = self.targets[idx]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img_np = np.array(img.convert("RGB"))

        if self.transform is not None:
            tensor = self.transform(image=img_np)["image"]
        else:
            tensor = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return tensor, target


def compute_ece_and_bins(probs, targets, n_bins=15):
    """
    Computes Expected Calibration Error (ECE) and bin-wise accuracy/confidence.
    probs: [N, C] numpy array of predicted probabilities
    targets: [N] numpy array of ground truth class indices
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_lower == 0.0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            gap = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * gap
            mce = max(mce, gap)

            bin_data.append({
                "bin_center": (bin_lower + bin_upper) / 2.0,
                "confidence": float(avg_confidence_in_bin),
                "accuracy": float(accuracy_in_bin),
                "weight": float(prop_in_bin),
                "count": int(np.sum(in_bin))
            })
        else:
            bin_data.append({
                "bin_center": (bin_lower + bin_upper) / 2.0,
                "confidence": 0.0,
                "accuracy": 0.0,
                "weight": 0.0,
                "count": 0
            })

    return float(ece), float(mce), bin_data


class ModelWithTemperature(nn.Module):
    """
    A thin decorator around a trained PyTorch model that learns a strictly
    positive scalar Temperature T via Platt scaling (Guo et al. 2017).
    Guarantees T > 0 by optimizing log(temperature).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.log_temperature = nn.Parameter(torch.zeros(1)) # exp(0.0) = 1.0

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        return logits / self.temperature

    def calibrate(self, val_loader, device="cpu"):
        """Extracts validation logits and tunes temperature parameter using LBFGS."""
        self.to(device)
        self.model.eval()
        logits_list = []
        labels_list = []

        print("[*] Extracting validation logits for calibration...")
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                logits = self.model(images)
                logits_list.append(logits.cpu())
                labels_list.append(targets)

        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)

        nll_criterion = nn.CrossEntropyLoss()

        # Measure pre-calibration NLL
        before_nll = nll_criterion(logits, labels).item()
        print(f"[*] Pre-calibration Validation NLL: {before_nll:.4f}")

        # Optimize log_temperature with L-BFGS with strong_wolfe line search
        optimizer = LBFGS([self.log_temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        after_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        optimal_t = float(self.temperature.item())
        print(f"[*] Post-calibration Validation NLL: {after_nll:.4f} (Reduction: {before_nll - after_nll:.4f})")
        print(f"[*] Optimal Learned Temperature (T): {optimal_t:.4f}")

        return optimal_t, logits.cpu().numpy(), labels.cpu().numpy()


def extract_logits_and_labels(model, data_loader, device="cpu"):
    """Extracts raw logits and labels from a dataset loader."""
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def plot_reliability_diagrams(val_uncal_bins, val_cal_bins, test_uncal_bins, test_cal_bins,
                             val_ece_pre, val_ece_post, test_ece_pre, test_ece_post,
                             temperature, save_path="reports/calibration_curves.png"):
    """Plots comparative 4-panel reliability diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Validation Set Reliability
    ax1 = axes[0]
    centers = [b["bin_center"] for b in val_uncal_bins if b["count"] > 0]
    uncal_accs = [b["accuracy"] for b in val_uncal_bins if b["count"] > 0]
    uncal_confs = [b["confidence"] for b in val_uncal_bins if b["count"] > 0]

    cal_centers = [b["bin_center"] for b in val_cal_bins if b["count"] > 0]
    cal_accs = [b["accuracy"] for b in val_cal_bins if b["count"] > 0]
    cal_confs = [b["confidence"] for b in val_cal_bins if b["count"] > 0]

    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
    ax1.plot(uncal_confs, uncal_accs, "s-", color="firebrick", label=f"Uncalibrated (T=1.0) | ECE: {val_ece_pre*100:.2f}%", linewidth=2)
    ax1.plot(cal_confs, cal_accs, "o-", color="forestgreen", label=f"Calibrated (T={temperature:.3f}) | ECE: {val_ece_post*100:.2f}%", linewidth=2)

    ax1.set_title("Validation Set Reliability Diagram (ECE Comparison)", fontsize=13, pad=10)
    ax1.set_xlabel("Confidence", fontsize=11)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="upper left")

    # Panel 2: Test Set Generalization Reliability
    ax2 = axes[1]
    t_centers = [b["bin_center"] for b in test_uncal_bins if b["count"] > 0]
    t_uncal_accs = [b["accuracy"] for b in test_uncal_bins if b["count"] > 0]
    t_uncal_confs = [b["confidence"] for b in test_uncal_bins if b["count"] > 0]

    t_cal_confs = [b["confidence"] for b in test_cal_bins if b["count"] > 0]
    t_cal_accs = [b["accuracy"] for b in test_cal_bins if b["count"] > 0]

    ax2.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)
    ax2.plot(t_uncal_confs, t_uncal_accs, "s-", color="firebrick", label=f"Test Uncalibrated | ECE: {test_ece_pre*100:.2f}%", linewidth=2)
    ax2.plot(t_cal_confs, t_cal_accs, "o-", color="forestgreen", label=f"Test Calibrated (T={temperature:.3f}) | ECE: {test_ece_post*100:.2f}%", linewidth=2)

    ax2.set_title("Held-Out Test Set Generalization Reliability Diagram", fontsize=13, pad=10)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Empirical Accuracy", fontsize=11)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10, loc="upper left")

    plt.suptitle("Phase 8: Temperature Scaling Calibration Analysis & Reliability Diagrams", fontsize=15, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[*] Calibration curves saved to {save_path}")


def run_calibration():
    print("=" * 80)
    print("PHASE 8: TEMPERATURE SCALING CONFIDENCE CALIBRATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # Load classes
    with open("data/supported_classes.json", "r") as f:
        supported_classes = json.load(f)
    num_classes = len(supported_classes)

    # Load trained model
    ckpt_path = "models/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "mobilenet_v3_large")
    base_model = build_model(model_name, num_classes=num_classes, pretrained=False)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.to(device)

    # Load manifests
    val_df = pd.read_csv("data/manifests/val.csv")
    test_df = pd.read_csv("data/manifests/test.csv")
    val_transform = get_val_transforms(img_size=224)

    val_loader = DataLoader(ManifestEvalDataset(val_df, transform=val_transform), batch_size=64, shuffle=False)
    test_loader = DataLoader(ManifestEvalDataset(test_df, transform=val_transform), batch_size=64, shuffle=False)

    # Calibrate with Temperature Scaling
    calibrator = ModelWithTemperature(base_model)
    calibrator.to(device)
    optimal_temperature, val_logits, val_targets = calibrator.calibrate(val_loader, device=device)

    # Extract test logits
    print("[*] Extracting test logits for generalization check...")
    test_logits, test_targets = extract_logits_and_labels(base_model, test_loader, device=device)

    # Calculate probabilities
    val_probs_uncal = np.exp(val_logits) / np.sum(np.exp(val_logits), axis=1, keepdims=True)
    val_probs_cal = np.exp(val_logits / optimal_temperature) / np.sum(np.exp(val_logits / optimal_temperature), axis=1, keepdims=True)

    test_probs_uncal = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)
    test_probs_cal = np.exp(test_logits / optimal_temperature) / np.sum(np.exp(test_logits / optimal_temperature), axis=1, keepdims=True)

    # Compute ECE & MCE
    val_ece_pre, val_mce_pre, val_uncal_bins = compute_ece_and_bins(val_probs_uncal, val_targets)
    val_ece_post, val_mce_post, val_cal_bins = compute_ece_and_bins(val_probs_cal, val_targets)

    test_ece_pre, test_mce_pre, test_uncal_bins = compute_ece_and_bins(test_probs_uncal, test_targets)
    test_ece_post, test_mce_post, test_cal_bins = compute_ece_and_bins(test_probs_cal, test_targets)

    print("\n" + "="*50)
    print("CALIBRATION METRICS SUMMARY")
    print("="*50)
    print(f"Optimal Temperature (T):        {optimal_temperature:.4f}")
    print(f"Validation ECE (Pre-scaling):   {val_ece_pre*100:.2f}%")
    print(f"Validation ECE (Post-scaling):  {val_ece_post*100:.2f}% (Reduction: {((val_ece_pre-val_ece_post)/val_ece_pre)*100:.1f}%)")
    print(f"Validation MCE (Post-scaling):  {val_mce_post*100:.2f}%")
    print(f"Test Set ECE (Pre-scaling):     {test_ece_pre*100:.2f}%")
    print(f"Test Set ECE (Post-scaling):    {test_ece_post*100:.2f}% (Reduction: {((test_ece_pre-test_ece_post)/test_ece_pre)*100:.1f}%)")
    print(f"Test Set MCE (Post-scaling):    {test_mce_post*100:.2f}%")

    # Plot reliability diagrams
    plot_reliability_diagrams(
        val_uncal_bins, val_cal_bins, test_uncal_bins, test_cal_bins,
        val_ece_pre, val_ece_post, test_ece_pre, test_ece_post,
        optimal_temperature, save_path="reports/calibration_curves.png"
    )

    # Save calibration parameters to models/calibration.json
    calibration_dict = {
        "temperature": round(float(optimal_temperature), 4),
        "method": "temperature_scaling",
        "validation": {
            "samples": len(val_targets),
            "uncalibrated_ece": round(float(val_ece_pre), 4),
            "calibrated_ece": round(float(val_ece_post), 4),
            "uncalibrated_mce": round(float(val_mce_pre), 4),
            "calibrated_mce": round(float(val_mce_post), 4),
            "relative_reduction_pct": round(float(((val_ece_pre - val_ece_post) / val_ece_pre) * 100), 2)
        },
        "test": {
            "samples": len(test_targets),
            "uncalibrated_ece": round(float(test_ece_pre), 4),
            "calibrated_ece": round(float(test_ece_post), 4),
            "uncalibrated_mce": round(float(test_mce_pre), 4),
            "calibrated_mce": round(float(test_mce_post), 4),
            "relative_reduction_pct": round(float(((test_ece_pre - test_ece_post) / test_ece_pre) * 100), 2)
        },
        "disclaimer": "The output is an AI-assisted indication, NOT a guaranteed agricultural diagnosis."
    }

    cal_path = "models/calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration_dict, f, indent=2)
    print(f"[*] Calibration parameters successfully serialized to {cal_path}")


if __name__ == "__main__":
    run_calibration()
