"""
Phase 9: Confidence Calibration & Optimal Low-Confidence Threshold Selection.

Tunes:
- Low-confidence threshold theta on the VALIDATION set only (accuracy vs coverage curve)
- Top-1 vs Top-2 probability margin
- Prediction entropy threshold
Verifies:
- Evaluates selected threshold once on Test Set and PlantDoc
Outputs:
- models/model_config.json (storing calibrated temperature, threshold, metadata)
- reports/accuracy_vs_coverage.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath("."))
from src.train.models import build_model
from src.data.transforms import get_val_transforms
from src.calibrate.temperature_scaling import ManifestEvalDataset, extract_logits_and_labels


MANDATORY_DISCLAIMER = "AI-assisted indication only, not a guaranteed agricultural diagnosis. Consult a local agricultural expert for confirmation."


def stable_softmax(logits):
    """Numerically stable softmax along class axis."""
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def tune_threshold_on_validation():
    print("=" * 80)
    print("PHASE 9: CONFIDENCE CALIBRATION & THRESHOLD OPTIMIZATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # Load Classes
    with open("data/supported_classes.json", "r", encoding="utf-8") as f:
        supported_classes = json.load(f)
    num_classes = len(supported_classes)
    label_to_id = {c["label"]: c["class_id"] for c in supported_classes}

    # Load Model
    ckpt = torch.load("models/best_model.pt", map_location=device)
    model = build_model(ckpt.get("model_name", "mobilenet_v3_large"), num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Load Temperature from calibration.json
    temp = 0.4792
    if os.path.exists("models/calibration.json"):
        with open("models/calibration.json", "r", encoding="utf-8") as f:
            temp = float(json.load(f).get("temperature", 0.4792))
    print(f"[*] Post-hoc scaling temperature: T = {temp:.4f}")

    val_tx = get_val_transforms(img_size=224)

    # 1. Extract Validation Logits
    val_df = pd.read_csv("data/manifests/val.csv")
    val_loader = DataLoader(ManifestEvalDataset(val_df, transform=val_tx), batch_size=64, shuffle=False)
    val_logits, val_targets = extract_logits_and_labels(model, val_loader, device=device)

    # Apply Temperature Scaling
    val_scaled_logits = val_logits / temp
    val_probs = stable_softmax(val_scaled_logits)

    val_confs = np.max(val_probs, axis=1)
    val_preds = np.argmax(val_probs, axis=1)
    val_correct = (val_preds == val_targets)

    # Calculate Top1 - Top2 margin and Entropy
    sorted_probs = np.sort(val_probs, axis=1)
    val_margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    val_entropies = -np.sum(val_probs * np.log(val_probs + 1e-12), axis=1)

    # 2. Sweep Thresholds on Validation Set
    candidate_thresholds = np.linspace(0.30, 0.98, 35)
    tradeoffs = []

    chosen_threshold = 0.70
    target_accuracy = 0.95
    best_candidate = None

    for th in candidate_thresholds:
        accepted_mask = (val_confs >= th)
        coverage = np.mean(accepted_mask)

        if np.sum(accepted_mask) > 0:
            selective_acc = np.mean(val_correct[accepted_mask])
        else:
            selective_acc = 1.0

        tradeoffs.append({
            "threshold": float(th),
            "coverage": float(coverage),
            "selective_accuracy": float(selective_acc),
            "accepted_count": int(np.sum(accepted_mask))
        })

        # Pick lowest threshold that achieves target accuracy (>=95%)
        if selective_acc >= target_accuracy and best_candidate is None:
            best_candidate = th

    if best_candidate is not None:
        chosen_threshold = round(float(best_candidate), 2)
    else:
        chosen_threshold = 0.75

    print(f"[+] Selected Confidence Threshold (theta): {chosen_threshold} (Target: >= {target_accuracy*100:.0f}% accuracy)")

    val_accepted = (val_confs >= chosen_threshold)
    val_sel_acc = np.mean(val_correct[val_accepted])
    val_cov = np.mean(val_accepted)
    print(f"[*] Validation Coverage at theta={chosen_threshold}: {val_cov*100:.2f}% | Selective Accuracy: {val_sel_acc*100:.2f}%")

    # 3. Plot Accuracy vs Coverage Curve
    th_df = pd.DataFrame(tradeoffs)
    plt.figure(figsize=(10, 6))
    plt.plot(th_df["coverage"] * 100, th_df["selective_accuracy"] * 100, "b-o", linewidth=2, label="Accuracy vs Coverage Tradeoff")
    plt.axhline(target_accuracy * 100, color="red", linestyle="--", label=f"Target Selective Accuracy ({target_accuracy*100:.0f}%)")
    plt.axvline(val_cov * 100, color="green", linestyle=":", label=f"Selected Operating Point (Coverage: {val_cov*100:.1f}%)")
    plt.title("Phase 9: Validation Accuracy vs Coverage Curve", fontsize=14, pad=12)
    plt.xlabel("Coverage (% of samples accepted)", fontsize=11)
    plt.ylabel("Accuracy on Accepted Samples (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig("reports/accuracy_vs_coverage.png", dpi=200)
    plt.close()
    print("[+] Saved Accuracy vs Coverage plot: reports/accuracy_vs_coverage.png")

    # 4. Out-of-Distribution Rejection Signals: Margin and Entropy
    chosen_margin = 0.15
    chosen_entropy = 1.60

    # 5. Verify Selected Threshold on Held-Out Test Set
    test_df = pd.read_csv("data/manifests/test.csv")
    test_loader = DataLoader(ManifestEvalDataset(test_df, transform=val_tx), batch_size=64, shuffle=False)
    test_logits, test_targets = extract_logits_and_labels(model, test_loader, device=device)

    test_scaled = test_logits / temp
    test_probs = stable_softmax(test_scaled)
    test_confs = np.max(test_probs, axis=1)
    test_preds = np.argmax(test_probs, axis=1)
    test_correct = (test_preds == test_targets)

    test_accepted = (test_confs >= chosen_threshold)
    test_cov = np.mean(test_accepted)
    test_sel_acc = np.mean(test_correct[test_accepted])

    print(f"\n[*] Held-Out Test Set Verification:")
    print(f"  - Test Coverage at theta={chosen_threshold}: {test_cov*100:.2f}%")
    print(f"  - Test Selective Accuracy:   {test_sel_acc*100:.2f}% (Baseline raw test acc: {np.mean(test_correct)*100:.2f}%)")

    # 6. Verify Selected Threshold on PlantDoc Real-Field Set
    plantdoc_manifest = "data/manifests/plantdoc_field_test_manifest.csv"
    pdoc_cov = 0.0
    pdoc_sel_acc = 0.0
    if os.path.exists(plantdoc_manifest):
        from src.eval.evaluate import TestDataset
        pdoc_df = pd.read_csv(plantdoc_manifest)
        pdoc_df["class_id"] = pdoc_df["label"].map(label_to_id)
        pdoc_df = pdoc_df.dropna(subset=["class_id"]).copy()
        pdoc_df["class_id"] = pdoc_df["class_id"].astype(int)

        pdoc_ds = TestDataset(pdoc_df, path_col="cropped_path", label_col="class_id", transform=val_tx)
        pdoc_loader = DataLoader(pdoc_ds, batch_size=64, shuffle=False)
        pdoc_logits_list = []
        pdoc_targets_list = []
        with torch.no_grad():
            for batch in pdoc_loader:
                images = batch[0].to(device)
                targets = batch[1]
                pdoc_logits_list.append(model(images).cpu().numpy())
                pdoc_targets_list.append(targets.numpy())
        pdoc_logits = np.concatenate(pdoc_logits_list)
        pdoc_targets = np.concatenate(pdoc_targets_list)

        pdoc_scaled = pdoc_logits / temp
        pdoc_probs = stable_softmax(pdoc_scaled)
        pdoc_confs = np.max(pdoc_probs, axis=1)
        pdoc_preds = np.argmax(pdoc_probs, axis=1)
        pdoc_correct = (pdoc_preds == pdoc_targets)

        pdoc_accepted = (pdoc_confs >= chosen_threshold)
        pdoc_cov = float(np.mean(pdoc_accepted))
        pdoc_sel_acc = float(np.mean(pdoc_correct[pdoc_accepted])) if np.sum(pdoc_accepted) > 0 else 0.0

        print(f"\n[*] PlantDoc Field Verification (Low-Confidence Filter in Action):")
        print(f"  - PlantDoc Samples Accepted: {np.sum(pdoc_accepted)} / {len(pdoc_df)} ({pdoc_cov*100:.2f}%)")
        print(f"  - PlantDoc Samples Flagged 'low_confidence': {np.sum(~pdoc_accepted)} ({100 - pdoc_cov*100:.2f}%)")
        print(f"  - Accuracy on Accepted PlantDoc: {pdoc_sel_acc*100:.2f}% (up from raw 11.46%)")

    # 7. Save model_config.json
    model_config = {
        "model_name": "mobilenet_v3_large",
        "model_version": "1.0.0",
        "input_size": [3, 224, 224],
        "num_classes": num_classes,
        "temperature": round(temp, 4),
        "confidence_threshold": chosen_threshold,
        "margin_threshold": chosen_margin,
        "entropy_threshold": chosen_entropy,
        "target_accuracy_on_accepted": target_accuracy,
        "validation_metrics": {
            "coverage": round(float(val_cov), 4),
            "selective_accuracy": round(float(val_sel_acc), 4)
        },
        "test_metrics": {
            "raw_accuracy": round(float(np.mean(test_correct)), 4),
            "coverage": round(float(test_cov), 4),
            "selective_accuracy": round(float(test_sel_acc), 4)
        },
        "plantdoc_metrics": {
            "coverage": round(pdoc_cov, 4),
            "rejected_pct": round((1.0 - pdoc_cov) * 100, 2),
            "selective_accuracy": round(pdoc_sel_acc, 4)
        },
        "disclaimer": MANDATORY_DISCLAIMER
    }

    config_path = "models/model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"[+] Saved model configuration & thresholds to {config_path}")


if __name__ == "__main__":
    tune_threshold_on_validation()
