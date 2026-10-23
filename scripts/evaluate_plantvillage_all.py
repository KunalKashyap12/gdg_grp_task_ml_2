"""
Comprehensive Evaluation Suite: Benchmarks production inference pipeline against 
PlantVillage dataset folders (20,636 images total across 15 classes).

Priority targets explicitly requested by user:
1. Potato___healthy (152 images)
2. Pepper__bell___healthy (1,477 images)
3. Full PlantVillage suite (all 15 supported foliar classes)
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import time
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.serve.inference import get_inference_engine
from src.data.transforms import get_val_transforms

FOLDER_TO_LABEL_MAP = {
    "Pepper__bell___Bacterial_spot": "Pepper___Bacterial_spot",
    "Pepper__bell___healthy": "Pepper___Healthy",
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___healthy": "Potato___Healthy",
    "Potato___Late_blight": "Potato___Late_blight",
    "Tomato_Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato_Early_blight": "Tomato___Early_blight",
    "Tomato_healthy": "Tomato___Healthy",
    "Tomato_Late_blight": "Tomato___Late_blight",
    "Tomato_Leaf_Mold": "Tomato___Leaf_mold",
    "Tomato_Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato___Spider_mites",
    "Tomato__Target_Spot": "Tomato___Target_spot",
    "Tomato__Tomato_mosaic_virus": "Tomato___Mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato___Yellow_leaf_curl_virus"
}


def preprocess_image_item(args):
    """Worker function for multithreaded image loading and augmentation."""
    path, transform, engine = args
    try:
        bgr = cv2.imread(path)
        if bgr is None or bgr.size == 0:
            return None, path, "corrupted"

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # Leaf Gatekeeper validation
        is_leaf, gatekeeper_status, _ = engine.validate_leaf_image(rgb)
        if not is_leaf:
            return None, path, gatekeeper_status

        # Multi-view TTA: Original + Horizontal Flip
        t1 = transform(image=rgb)["image"]
        flipped = np.ascontiguousarray(np.fliplr(rgb))
        t2 = transform(image=flipped)["image"]

        stacked = torch.stack([t1, t2], dim=0) # [2, 3, 224, 224]
        return stacked, path, "valid"
    except Exception as e:
        return None, path, f"error: {str(e)}"


def evaluate_single_class_folder(folder_path, expected_label, engine, batch_size=32, num_threads=8):
    """Evaluates an entire single class directory with batched ONNX execution."""
    folder_name = os.path.basename(folder_path)
    expected_meta = next(c for c in engine.classes_list if c["label"] == expected_label)
    expected_crop = expected_meta["crop"]
    expected_disease = expected_meta["disease"]

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    total_files = len(files)

    if total_files == 0:
        return {"folder": folder_name, "error": "No image files found"}

    transform = get_val_transforms(img_size=224, method="center_crop")
    
    correct_top1 = 0
    correct_top3 = 0
    correct_crop = 0
    correct_disease = 0
    accepted_correct_top1 = 0
    
    status_counts = Counter()
    predicted_labels = Counter()
    confidences = []
    latencies = []

    t_folder_start = time.perf_counter()

    # Process in batches
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for b_start in range(0, total_files, batch_size):
            b_files = files[b_start:b_start + batch_size]
            tasks = [(p, transform, engine) for p in b_files]
            items = list(executor.map(preprocess_image_item, tasks))

            valid_tensors = []
            valid_paths = []
            
            for tensor, path, status in items:
                if status == "valid" and tensor is not None:
                    valid_tensors.append(tensor)
                    valid_paths.append(path)
                else:
                    status_counts[status] += 1
                    predicted_labels["Gatekeeper_Rejected"] += 1

            if not valid_tensors:
                continue

            # Batch dimensions: [N, 2, 3, 224, 224] -> [N * 2, 3, 224, 224]
            N = len(valid_tensors)
            batch_tensor = torch.stack(valid_tensors, dim=0) # [N, 2, 3, 224, 224]
            flat_tensor = batch_tensor.view(N * 2, 3, 224, 224).numpy().astype(np.float32)

            t0 = time.perf_counter()
            ort_out = engine.session.run([engine.output_name], {engine.input_name: flat_tensor})[0]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / N

            # Average logits across TTA views
            logits = ort_out.reshape(N, 2, -1).mean(axis=1) # [N, 37]
            cal_logits = logits / engine.temperature
            cal_probs = engine.softmax(cal_logits)

            for i in range(N):
                latencies.append(elapsed_ms)
                probs_i = cal_probs[i]
                raw_logits_i = logits[i]

                # Stage 1: Marginal posterior crop probability
                crop_probs = {c: float(np.sum(probs_i[indices])) for c, indices in engine.crop_to_indices.items()}
                top_crop = max(crop_probs, key=crop_probs.get)

                # Stage 2: Foliar disease probability within top crop
                active_indices = engine.crop_to_indices[top_crop]
                sub_logits = raw_logits_i[active_indices] / engine.temperature
                sub_probs = engine.softmax(sub_logits)
                best_sub_rank = int(np.argmax(sub_probs))
                best_idx = active_indices[best_sub_rank]
                
                pred_meta = engine.class_map[best_idx]
                pred_label = pred_meta["label"]
                pred_crop = pred_meta["crop"]
                pred_disease = pred_meta["disease"]

                # Global calibrated confidence and OOD entropy check
                final_conf = float(probs_i[best_idx])
                confidences.append(final_conf)
                predicted_labels[pred_label] += 1

                entropy = -float(np.sum(probs_i * np.log(probs_i + 1e-12)))
                is_low_conf = (final_conf < engine.threshold) or (entropy > engine.entropy_threshold)
                status = "low_confidence" if is_low_conf else "success"
                status_counts[status] += 1

                # Top-3 global predictions
                top3_indices = np.argsort(probs_i)[::-1][:3]
                top3_labels = [engine.class_map[int(idx)]["label"] for idx in top3_indices]

                # Match evaluations
                is_top1 = (pred_label == expected_label)
                is_top3 = (expected_label in top3_labels)
                is_crop_match = (pred_crop.lower() == expected_crop.lower())
                is_disease_match = (pred_disease.lower() == expected_disease.lower())

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

    total_time = time.perf_counter() - t_folder_start
    accepted_count = status_counts["success"]

    metrics = {
        "folder": folder_name,
        "ground_truth_label": expected_label,
        "crop": expected_crop,
        "disease": expected_disease,
        "total_images": total_files,
        "top1_correct": correct_top1,
        "top1_accuracy_pct": round(correct_top1 / total_files * 100.0, 2),
        "top3_accuracy_pct": round(correct_top3 / total_files * 100.0, 2),
        "crop_accuracy_pct": round(correct_crop / total_files * 100.0, 2),
        "disease_accuracy_pct": round(correct_disease / total_files * 100.0, 2),
        "selective_accuracy_pct": round(accepted_correct_top1 / accepted_count * 100.0, 2) if accepted_count > 0 else 0.0,
        "status_breakdown": {
            "success_accepted": accepted_count,
            "low_confidence": status_counts["low_confidence"],
            "not_a_leaf": status_counts["not_a_leaf"]
        },
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
        "median_confidence": round(float(np.median(confidences)), 4) if confidences else 0.0,
        "elapsed_sec": round(total_time, 2),
        "throughput_fps": round(total_files / total_time, 1),
        "top_predictions": predicted_labels.most_common(4)
    }

    return metrics


def run_full_suite():
    pv_root = r"c:\Users\HP\OneDrive\Desktop\gdg-grp-task(ml)\data\raw\PlantVillage\PlantVillage"
    print("=" * 70)
    print("FLORASCAN AI - FULL PLANTVILLAGE DATASET BENCHMARK")
    print(f"Target Root: {pv_root}")
    print("=" * 70)

    engine = get_inference_engine()

    # Priority target evaluation sequence:
    # 1. Potato___healthy (Explicit user request)
    # 2. Pepper__bell___healthy (Explicit user request)
    # 3. All remaining classes in PlantVillage
    priority_order = [
        "Potato___healthy",
        "Pepper__bell___healthy"
    ]
    other_folders = [f for f in FOLDER_TO_LABEL_MAP.keys() if f not in priority_order]
    execution_list = priority_order + other_folders

    folder_results = []
    total_imgs_all = 0
    total_top1_all = 0
    total_top3_all = 0
    total_accepted_all = 0
    total_accepted_correct_all = 0

    t_suite_start = time.perf_counter()

    for idx, folder_name in enumerate(execution_list, 1):
        folder_path = os.path.join(pv_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"[-] Skipping missing folder: {folder_name}")
            continue

        label = FOLDER_TO_LABEL_MAP[folder_name]
        print(f"\n[{idx}/{len(execution_list)}] Evaluating: {folder_name} (Canonical: {label})...")
        
        res = evaluate_single_class_folder(folder_path, label, engine, batch_size=32, num_threads=8)
        folder_results.append(res)

        total_imgs_all += res["total_images"]
        total_top1_all += res["top1_correct"]
        total_top3_all += int(res["total_images"] * (res["top3_accuracy_pct"] / 100.0))
        total_accepted_all += res["status_breakdown"]["success_accepted"]
        total_accepted_correct_all += int(res["status_breakdown"]["success_accepted"] * (res["selective_accuracy_pct"] / 100.0))

        print(f"  -> Total Images:    {res['total_images']}")
        print(f"  -> Top-1 Accuracy:  {res['top1_accuracy_pct']}% ({res['top1_correct']}/{res['total_images']})")
        print(f"  -> Top-3 Accuracy:  {res['top3_accuracy_pct']}%")
        print(f"  -> Crop Accuracy:   {res['crop_accuracy_pct']}%")
        print(f"  -> Disease Match:   {res['disease_accuracy_pct']}%")
        print(f"  -> Selective Acc:   {res['selective_accuracy_pct']}% (on accepted conf >= 0.74)")
        print(f"  -> Mean / Median:   {res['mean_confidence']*100:.1f}% / {res['median_confidence']*100:.1f}%")
        print(f"  -> Throughput:      {res['throughput_fps']} img/sec ({res['elapsed_sec']}s)")

    total_suite_sec = time.perf_counter() - t_suite_start

    macro_top1 = np.mean([r["top1_accuracy_pct"] for r in folder_results])
    macro_top3 = np.mean([r["top3_accuracy_pct"] for r in folder_results])
    micro_top1 = (total_top1_all / total_imgs_all * 100.0) if total_imgs_all > 0 else 0.0
    overall_selective = (total_accepted_correct_all / total_accepted_all * 100.0) if total_accepted_all > 0 else 0.0

    suite_summary = {
        "dataset": "PlantVillage",
        "total_classes_tested": len(folder_results),
        "total_images_tested": total_imgs_all,
        "macro_avg_top1_pct": round(float(macro_top1), 2),
        "micro_avg_top1_pct": round(float(micro_top1), 2),
        "macro_avg_top3_pct": round(float(macro_top3), 2),
        "overall_selective_accuracy_pct": round(float(overall_selective), 2),
        "overall_accepted_count": total_accepted_all,
        "total_runtime_sec": round(total_suite_sec, 2),
        "overall_throughput_fps": round(total_imgs_all / total_suite_sec, 1),
        "folder_metrics": folder_results
    }

    out_file = "reports/eval_plantvillage_all_results.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("GLOBAL PLANTVILLAGE BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Total Images Evaluated: {total_imgs_all}")
    print(f"Micro-Average Top-1:   {micro_top1:.2f}%")
    print(f"Macro-Average Top-1:   {macro_top1:.2f}%")
    print(f"Macro-Average Top-3:   {macro_top3:.2f}%")
    print(f"Selective Accuracy:    {overall_selective:.2f}% (High-confidence accepted)")
    print(f"Total Time:            {total_suite_sec:.2f}s ({total_imgs_all/total_suite_sec:.1f} images/sec)")
    print(f"[+] Full JSON report saved to: {out_file}")


if __name__ == "__main__":
    run_full_suite()
