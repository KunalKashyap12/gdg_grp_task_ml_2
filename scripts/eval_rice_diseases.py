"""
Evaluation Script: Evaluates Kaggle_Rice (120 images across 3 folders)
- Bacterial leaf blight (40 images) -> Supported: Rice___Bacterial_leaf_blight
- Brown spot (40 images)           -> Supported: Rice___Brown_spot
- Leaf smut (40 images)            -> Out-of-Distribution (OOD) / Unsupported Rice Disease
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import time
import json
from collections import Counter
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.serve.inference import get_inference_engine
from src.data.transforms import get_val_transforms

def evaluate_rice_folder(folder_path, expected_label, is_supported=True):
    engine = get_inference_engine()
    transform = get_val_transforms(img_size=224, method="center_crop")
    
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    total_files = len(files)
    folder_name = os.path.basename(folder_path)

    print(f"\n[*] Evaluating Rice Category: {folder_name} ({total_files} images)...")
    print(f"    Expected Class: {expected_label} (Supported: {is_supported})")

    expected_crop = "Rice"
    expected_disease = expected_label.split("___")[-1] if is_supported else "Leaf_smut"

    correct_top1 = 0
    correct_top3 = 0
    correct_crop = 0
    correct_disease = 0
    accepted_correct_top1 = 0
    
    status_counts = Counter()
    predicted_labels = Counter()
    predicted_crops = Counter()
    confidences = []
    sample_diagnoses = []

    t_start = time.perf_counter()

    for idx, path in enumerate(files, 1):
        bgr = cv2.imread(path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Gatekeeper
        is_leaf, gatekeeper_status, msg = engine.validate_leaf_image(rgb)
        if not is_leaf:
            status_counts[gatekeeper_status] += 1
            predicted_labels["Not_a_leaf"] += 1
            continue

        # Preprocessing with TTA: Original + Horizontal Flip
        t1 = transform(image=rgb)["image"]
        flipped = np.ascontiguousarray(np.fliplr(rgb))
        t2 = transform(image=flipped)["image"]
        batch_t = torch.stack([t1, t2], dim=0).numpy().astype(np.float32)

        # ONNX inference
        ort_out = engine.session.run([engine.output_name], {engine.input_name: batch_t})[0]
        logits = np.mean(ort_out, axis=0)
        cal_logits = logits / engine.temperature
        probs = engine.softmax(cal_logits)

        # Stage 1: Crop identification
        crop_probs = {c: float(np.sum(probs[indices])) for c, indices in engine.crop_to_indices.items()}
        sorted_crops = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)
        top_crop, top_crop_prob = sorted_crops[0]
        predicted_crops[top_crop] += 1

        # Stage 2: Foliar disease within top crop
        active_indices = engine.crop_to_indices[top_crop]
        sub_logits = logits[active_indices] / engine.temperature
        sub_probs = engine.softmax(sub_logits)
        best_sub_rank = int(np.argmax(sub_probs))
        best_idx = active_indices[best_sub_rank]
        
        pred_meta = engine.class_map[best_idx]
        pred_label = pred_meta["label"]
        pred_crop = pred_meta["crop"]
        pred_disease = pred_meta["disease"]
        conf = float(probs[best_idx])
        confidences.append(conf)
        predicted_labels[pred_label] += 1

        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        is_low_conf = (conf < engine.threshold) or (entropy > engine.entropy_threshold)
        status = "low_confidence" if is_low_conf else "success"
        status_counts[status] += 1

        top3_indices = np.argsort(probs)[::-1][:3]
        top3_labels = [engine.class_map[int(i)]["label"] for i in top3_indices]

        # Metric flags
        is_top1 = (pred_label == expected_label) if is_supported else False
        is_top3 = (expected_label in top3_labels) if is_supported else False
        is_crop_match = (pred_crop.lower() == expected_crop.lower())
        is_disease_match = (pred_disease.lower() == expected_disease.lower()) if is_supported else False

        if is_top1:
            correct_top1 += 1
        if is_top3:
            correct_top3 += 1
        if is_crop_match:
            correct_crop += 1
        if is_disease_match:
            correct_disease += 1
        if status == "success" and is_top1:
            accepted_correct_top1 += 1

        if len(sample_diagnoses) < 5 or (not is_top1 and len(sample_diagnoses) < 10):
            sample_diagnoses.append({
                "file": os.path.basename(path),
                "predicted_label": pred_label,
                "confidence": round(conf, 4),
                "status": status,
                "stage1_crop": top_crop,
                "top3": top3_labels
            })

    total_time = time.perf_counter() - t_start
    accepted_count = status_counts["success"]

    metrics = {
        "folder": folder_name,
        "is_supported_class": is_supported,
        "ground_truth_label": expected_label,
        "total_images": total_files,
        "top1_accuracy_pct": round(correct_top1 / total_files * 100.0, 2) if is_supported else None,
        "top3_accuracy_pct": round(correct_top3 / total_files * 100.0, 2) if is_supported else None,
        "crop_accuracy_pct": round(correct_crop / total_files * 100.0, 2),
        "disease_accuracy_pct": round(correct_disease / total_files * 100.0, 2) if is_supported else None,
        "selective_accuracy_pct": round(accepted_correct_top1 / accepted_count * 100.0, 2) if (is_supported and accepted_count > 0) else None,
        "status_breakdown": {
            "success_accepted": accepted_count,
            "low_confidence_flagged": status_counts["low_confidence"],
            "not_a_leaf": status_counts["not_a_leaf"]
        },
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "median_confidence": round(float(np.median(confidences)), 4) if confidences else 0.0,
        "elapsed_sec": round(total_time, 2),
        "predicted_crops_distribution": dict(predicted_crops),
        "top_predictions": predicted_labels.most_common(5),
        "sample_diagnoses": sample_diagnoses
    }

    print(f"  [+] Top-1 Accuracy:  {metrics['top1_accuracy_pct'] if is_supported else 'N/A (OOD Class)'}")
    print(f"  [+] Crop Accuracy:   {metrics['crop_accuracy_pct']}% (Identified as Rice)")
    print(f"  [+] Status Breakdown: Accepted: {accepted_count}, Low Conf: {status_counts['low_confidence']}")
    print(f"  [+] Top Predictions: {metrics['top_predictions']}")

    return metrics


def run_rice_evaluation():
    rice_root = r"c:\Users\HP\OneDrive\Desktop\gdg-grp-task(ml)\data\raw\Kaggle_Rice\rice_leaf_diseases"
    
    categories = [
        {"folder": "Bacterial leaf blight", "label": "Rice___Bacterial_leaf_blight", "supported": True},
        {"folder": "Brown spot", "label": "Rice___Brown_spot", "supported": True},
        {"folder": "Leaf smut", "label": "Unsupported_Rice_Leaf_Smut", "supported": False}
    ]

    all_metrics = []
    for cat in categories:
        fp = os.path.join(rice_root, cat["folder"])
        res = evaluate_rice_folder(fp, cat["label"], is_supported=cat["supported"])
        all_metrics.append(res)

    out_file = "reports/eval_rice_diseases_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[+] Full Rice Benchmark saved to: {out_file}")

if __name__ == "__main__":
    run_rice_evaluation()
