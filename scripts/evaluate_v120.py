"""
Evaluation and benchmark script for Optuna-tuned Model v1.2.0:
1. Lab test set: data/manifests/test.csv (N=2,829)
2. Field test set: data/manifests/field_test_heldout.csv (N=323)
3. False-healthy rates (overall, Potato, Rice)
4. Novel Out-of-Scope set: reports/novel_ood_manifest.csv (N=107 images)
5. Real photos: real_test/ (N=20 images)
6. Wilson 95% Confidence Intervals for all accuracies
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, ".")
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.serve.inference import CropDiseaseInferenceEngine

def wilson_score_interval(successes, total, confidence=0.95):
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    z = 1.95996
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p_hat + z**2 / (2 * total)) / denominator
    half_width = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    ci_lower = max(0.0, centre_adjusted_probability - half_width)
    ci_upper = min(1.0, centre_adjusted_probability + half_width)
    return p_hat, ci_lower, ci_upper

def get_albumentations_tensor(path, img_size=224):
    tx = A.Compose([
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=img_size, width=img_size, p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])
    full_path = resolve_image_path(path)
    with Image.open(full_path) as img:
        img = ImageOps.exif_transpose(img)
        img_np = np.array(img.convert("RGB"))
    tensor = tx(image=img_np)["image"].unsqueeze(0).numpy()
    return tensor

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def run_eval(session, config_data, df, name="split"):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    T = float(config_data.get("temperature", 1.0))
    threshold = float(config_data.get("confidence_threshold", 0.70))
    
    preds, confs, targets = [], [], []
    for _, row in df.iterrows():
        tensor = get_albumentations_tensor(row["filepath"])
        logits = session.run([output_name], {input_name: tensor})[0][0]
        cal_logits = logits / T
        cal_probs = softmax(cal_logits)
        
        pred_class = int(np.argmax(cal_probs[:37]))
        conf = float(cal_probs[pred_class])
        
        preds.append(pred_class)
        confs.append(conf)
        targets.append(int(row["class_id"]))
        
    preds = np.array(preds)
    confs = np.array(confs)
    targets = np.array(targets)
    
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    
    accepted = (confs >= threshold)
    cov = float(np.mean(accepted))
    sel_acc = float(np.mean(preds[accepted] == targets[accepted])) if np.sum(accepted) > 0 else 1.0
    
    p_hat, ci_l, ci_u = wilson_score_interval(np.sum(preds == targets), len(targets))
    
    return {
        "accuracy": acc,
        "ci_lower": ci_l,
        "ci_upper": ci_u,
        "macro_f1": macro_f1,
        "coverage": cov,
        "selective_accuracy": sel_acc,
        "preds": preds,
        "confs": confs,
        "targets": targets
    }

def main():
    print("=" * 75)
    print("[*] Benchmark Evaluation of Optuna-Tuned Model v1.2.0")
    print("=" * 75)
    
    model_dir = "models_v1.2.0"
    onnx_path = os.path.join(model_dir, "crop_disease_model.onnx")
    config_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(onnx_path) or not os.path.exists(config_path):
        print(f"[!] Error: Model artifacts not found in {model_dir}")
        return
        
    with open(config_path, "r") as f:
        config_data = json.load(f)
        
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[*] Loaded ONNX model from {onnx_path}")
    print(f"[*] Configuration: T = {config_data.get('temperature')}, threshold = {config_data.get('confidence_threshold')}")
    
    lab_df = pd.read_csv("data/manifests/test.csv")
    field_df = pd.read_csv("data/manifests/field_test_heldout.csv")
    ood_manifest_path = "reports/novel_ood_manifest.csv"
    if not os.path.exists(ood_manifest_path):
        ood_manifest_path = "scratch/novel_ood_manifest.csv"
    novel_ood_df = pd.read_csv(ood_manifest_path)
    
    print(f"[*] Datasets: Lab={len(lab_df)}, Field={len(field_df)}, Novel OOD={len(novel_ood_df)}")
    
    # 1. Lab Test
    print("\n[1] Evaluating Lab Test Set...")
    lab_res = run_eval(session, config_data, lab_df, name="Lab")
    print(f"  Accuracy:           {lab_res['accuracy']*100:.2f}% (95% CI: [{lab_res['ci_lower']*100:.2f}%, {lab_res['ci_upper']*100:.2f}%])")
    print(f"  Macro F1:           {lab_res['macro_f1']*100:.2f}%")
    print(f"  Coverage (th=0.70): {lab_res['coverage']*100:.2f}%")
    print(f"  Selective Accuracy: {lab_res['selective_accuracy']*100:.2f}%")
    
    # 2. Field Test
    print("\n[2] Evaluating Field Test Set...")
    field_res = run_eval(session, config_data, field_df, name="Field")
    print(f"  Accuracy:           {field_res['accuracy']*100:.2f}% (95% CI: [{field_res['ci_lower']*100:.2f}%, {field_res['ci_upper']*100:.2f}%])")
    print(f"  Macro F1:           {field_res['macro_f1']*100:.2f}%")
    print(f"  Coverage (th=0.70): {field_res['coverage']*100:.2f}%")
    print(f"  Selective Accuracy: {field_res['selective_accuracy']*100:.2f}%")
    
    # 3. False-Healthy Rates
    with open("data/supported_classes.json", "r") as f:
        classes_info = json.load(f)
    healthy_ids = {c["class_id"] for c in classes_info if c["disease"].lower() == "healthy"}
    potato_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Potato___Healthy")
    potato_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Potato" and c["disease"].lower() != "healthy"}
    rice_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Rice___Healthy")
    rice_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Rice" and c["disease"].lower() != "healthy"}
    
    lab_targets = lab_res["targets"]
    lab_preds = lab_res["preds"]
    
    diseased_mask = np.isin(lab_targets, list(healthy_ids), invert=True)
    diseased_preds = lab_preds[diseased_mask]
    overall_false_healthy = float(np.mean(np.isin(diseased_preds, list(healthy_ids))))
    
    potato_mask = np.isin(lab_targets, list(potato_diseased_ids))
    potato_false_healthy = float(np.mean(lab_preds[potato_mask] == potato_healthy_id)) if np.sum(potato_mask) > 0 else 0.0
    
    rice_mask = np.isin(lab_targets, list(rice_diseased_ids))
    rice_false_healthy = float(np.mean(lab_preds[rice_mask] == rice_healthy_id)) if np.sum(rice_mask) > 0 else 0.0
    
    print("\n[3] False-Healthy Error Rates (Lab Test Set):")
    print(f"  Overall False-Healthy: {overall_false_healthy*100:.2f}%")
    print(f"  Potato False-Healthy:  {potato_false_healthy*100:.2f}%")
    print(f"  Rice False-Healthy:    {rice_false_healthy*100:.2f}%")
    
    # 4. Novel Out-of-Scope Set
    print(f"\n[4] Evaluating Novel Out-of-Scope Rejection Benchmark (N={len(novel_ood_df)})...")
    os.environ["MODEL_DIR"] = model_dir
    engine = CropDiseaseInferenceEngine(
        onnx_model_path=onnx_path,
        config_path=config_path
    )
    
    ood_results = []
    for _, row in novel_ood_df.iterrows():
        with open(row["filepath"], "rb") as f:
            img_bytes = f.read()
        res = engine.predict_v1(img_bytes, crop_hint=None, top_k=3)
        status = res.get("status")
        conf = res.get("prediction", {}).get("confidence", 0.0) if res.get("prediction") else 0.0
        label = res.get("prediction", {}).get("label") if res.get("prediction") else None
        ood_results.append({
            "filepath": row["filepath"],
            "description": row["description"],
            "category": row["category"],
            "status": status,
            "confidence": conf,
            "label": label,
            "is_safely_flagged": (status in ["unsupported_crop", "low_confidence", "not_a_leaf"])
        })
        
    df_ood = pd.DataFrame(ood_results)
    safe_count = df_ood["is_safely_flagged"].sum()
    safe_rate = safe_count / len(df_ood)
    print(f"  Safely Flagged (unsupported_crop / low_confidence / not_a_leaf): {safe_count}/{len(df_ood)} ({safe_rate*100:.2f}%)")
    print(f"  Status breakdown:")
    for status, count in df_ood["status"].value_counts().items():
        print(f"    - {status}: {count} ({count/len(df_ood)*100:.1f}%)")
        
    # 5. Real Photos Evaluation
    real_test_dir = "real_test"
    if not os.path.exists(real_test_dir):
        real_test_dir = "data/real_test"
    real_images = sorted([f for f in os.listdir(real_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n[5] Evaluating Real Photos (N={len(real_images)} in {real_test_dir})...")
    
    real_results = []
    for fname in real_images:
        fpath = os.path.join(real_test_dir, fname)
        with open(fpath, "rb") as f:
            img_bytes = f.read()
        res = engine.predict_v1(img_bytes, crop_hint=None, top_k=3)
        status = res.get("status")
        pred_label = res.get("prediction", {}).get("label") if res.get("prediction") else None
        conf = res.get("prediction", {}).get("confidence", 0.0) if res.get("prediction") else 0.0
        real_results.append({
            "filename": fname,
            "status": status,
            "prediction": pred_label,
            "confidence": conf
        })
        print(f"    {fname:<25} -> status: {status:<18} | pred: {str(pred_label):<30} | conf: {conf:.4f}")
        
    # 6. Save Summary JSON
    summary_out = {
        "model_version": "1.2.0",
        "temperature": config_data.get("temperature"),
        "confidence_threshold": config_data.get("confidence_threshold"),
        "lab_test": {
            "n": len(lab_df),
            "accuracy": lab_res["accuracy"],
            "ci_lower": lab_res["ci_lower"],
            "ci_upper": lab_res["ci_upper"],
            "macro_f1": lab_res["macro_f1"],
            "coverage": lab_res["coverage"],
            "selective_accuracy": lab_res["selective_accuracy"]
        },
        "field_test": {
            "n": len(field_df),
            "accuracy": field_res["accuracy"],
            "ci_lower": field_res["ci_lower"],
            "ci_upper": field_res["ci_upper"],
            "macro_f1": field_res["macro_f1"],
            "coverage": field_res["coverage"],
            "selective_accuracy": field_res["selective_accuracy"]
        },
        "false_healthy": {
            "overall": overall_false_healthy,
            "potato": potato_false_healthy,
            "rice": rice_false_healthy
        },
        "novel_ood": {
            "n": len(df_ood),
            "safely_flagged_count": int(safe_count),
            "safely_flagged_rate": float(safe_rate),
            "status_breakdown": df_ood["status"].value_counts().to_dict()
        },
        "real_photos": real_results
    }
    
    with open("reports/improvement/eval_v120_results.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print("\n[+] Saved evaluation results to reports/improvement/eval_v120_results.json")

if __name__ == "__main__":
    main()
