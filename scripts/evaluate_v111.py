"""
Evaluation and benchmark script for Model v1.1.1:
1. Lab test set: data/manifests/test.csv (N=2,829)
2. Field test set: data/manifests/field_test_heldout.csv (N=323)
3. False-healthy rates (overall, Potato, Rice)
4. Novel Out-of-Scope set: scratch/novel_ood_manifest.csv (N=107 images)
5. Comparison against v1.1.0 and acceptance check
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score, recall_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

sys.path.insert(0, ".")
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD
from src.serve.inference import CropDiseaseInferenceEngine

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

def get_pil_tensor(path, img_size=224):
    full_path = resolve_image_path(path)
    with Image.open(full_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        w, h = img.size
        short = min(w, h)
        scale = 256.0 / short
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = (new_w - img_size) // 2
        top = (new_h - img_size) // 2
        crop_img = img_resized.crop((left, top, left + img_size, top + img_size))
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return arr.transpose(2, 0, 1)[np.newaxis, ...]

def softmax(logits):
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

def run_eval(session, config_data, df, preprocess_fn, name="split"):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    T = float(config_data.get("temperature", 1.0))
    threshold = float(config_data.get("confidence_threshold", 0.70))
    
    preds, confs, targets = [], [], []
    for _, row in df.iterrows():
        tensor = preprocess_fn(row["filepath"])
        logits = session.run([output_name], {input_name: tensor})[0][0]
        cal_logits = logits / T
        cal_probs = softmax(cal_logits)
        
        # Predicted class among 37 supported disease classes
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
    
    # Selective accuracy at threshold
    accepted = (confs >= threshold)
    cov = float(np.mean(accepted))
    sel_acc = float(np.mean(preds[accepted] == targets[accepted])) if np.sum(accepted) > 0 else 1.0
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "coverage": cov,
        "selective_accuracy": sel_acc,
        "preds": preds,
        "confs": confs,
        "targets": targets
    }

def main():
    print("=" * 70)
    print("[*] Evaluating Model v1.1.1 on Held-out Benchmarks")
    print("=" * 70)
    
    model_dir = "models_v1.1.1"
    onnx_path = os.path.join(model_dir, "crop_disease_model.onnx")
    config_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(onnx_path) or not os.path.exists(config_path):
        print(f"[!] Model files not found in {model_dir}")
        return
        
    with open(config_path, "r") as f:
        config_data = json.load(f)
        
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[*] Loaded ONNX model from {onnx_path}")
    print(f"[*] Configuration: T = {config_data.get('temperature')}, threshold = {config_data.get('confidence_threshold')}")
    
    # 1. Load Data
    lab_df = pd.read_csv("data/manifests/test.csv")
    field_df = pd.read_csv("data/manifests/field_test_heldout.csv")
    novel_ood_df = pd.read_csv("scratch/novel_ood_manifest.csv")
    
    print(f"[*] Datasets: Lab={len(lab_df)}, Field={len(field_df)}, Novel OOD={len(novel_ood_df)}")
    
    # 2. Evaluate with Albumentations (Unified Production Pipeline)
    print("\n--- [A] Unified Albumentations Pipeline Evaluation ---")
    lab_res_alb = run_eval(session, config_data, lab_df, get_albumentations_tensor, name="Lab (Alb)")
    print(f"  Lab Test Set (N={len(lab_df)}):")
    print(f"    Raw Accuracy:       {lab_res_alb['accuracy']*100:.2f}%")
    print(f"    Macro F1:           {lab_res_alb['macro_f1']*100:.2f}%")
    print(f"    Coverage (th={config_data['confidence_threshold']}):  {lab_res_alb['coverage']*100:.2f}%")
    print(f"    Selective Accuracy: {lab_res_alb['selective_accuracy']*100:.2f}%")
    
    field_res_alb = run_eval(session, config_data, field_df, get_albumentations_tensor, name="Field (Alb)")
    print(f"  Field Test Set (N={len(field_df)}):")
    print(f"    Raw Accuracy:       {field_res_alb['accuracy']*100:.2f}%")
    print(f"    Macro F1:           {field_res_alb['macro_f1']*100:.2f}%")
    print(f"    Coverage (th={config_data['confidence_threshold']}):  {field_res_alb['coverage']*100:.2f}%")
    print(f"    Selective Accuracy: {field_res_alb['selective_accuracy']*100:.2f}%")

    # 3. False-Healthy Rates
    with open("data/supported_classes.json", "r") as f:
        classes_info = json.load(f)
    healthy_ids = {c["class_id"] for c in classes_info if c["disease"].lower() == "healthy"}
    potato_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Potato___Healthy")
    potato_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Potato" and c["disease"].lower() != "healthy"}
    rice_healthy_id = next(c["class_id"] for c in classes_info if c["label"] == "Rice___Healthy")
    rice_diseased_ids = {c["class_id"] for c in classes_info if c["crop"] == "Rice" and c["disease"].lower() != "healthy"}
    
    lab_targets = lab_res_alb["targets"]
    lab_preds = lab_res_alb["preds"]
    
    diseased_mask = np.isin(lab_targets, list(healthy_ids), invert=True)
    diseased_preds = lab_preds[diseased_mask]
    overall_false_healthy = float(np.mean(np.isin(diseased_preds, list(healthy_ids))))
    
    potato_mask = np.isin(lab_targets, list(potato_diseased_ids))
    potato_false_healthy = float(np.mean(lab_preds[potato_mask] == potato_healthy_id)) if np.sum(potato_mask) > 0 else 0.0
    
    rice_mask = np.isin(lab_targets, list(rice_diseased_ids))
    rice_false_healthy = float(np.mean(lab_preds[rice_mask] == rice_healthy_id)) if np.sum(rice_mask) > 0 else 0.0
    
    print("\n  False-Healthy Error Rates (Lab Test Set):")
    print(f"    Overall False-Healthy: {overall_false_healthy*100:.2f}%")
    print(f"    Potato False-Healthy:  {potato_false_healthy*100:.2f}%")
    print(f"    Rice False-Healthy:    {rice_false_healthy*100:.2f}%")

    # 4. Novel Out-of-Scope Evaluation (N=107) using Production CropDiseaseInferenceEngine
    print(f"\n--- [B] Novel Out-of-Scope Rejection Benchmark (N={len(novel_ood_df)} images) ---")
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
        
    df_ood_res = pd.DataFrame(ood_results)
    safe_count = int(df_ood_res["is_safely_flagged"].sum())
    total_ood = len(df_ood_res)
    rejection_rate = safe_count / total_ood
    unsupp_count = int((df_ood_res["status"] == "unsupported_crop").sum())
    low_conf_count = int((df_ood_res["status"] == "low_confidence").sum())
    not_leaf_count = int((df_ood_res["status"] == "not_a_leaf").sum())
    leak_count = int((df_ood_res["status"] == "success").sum())
    
    print(f"  Safely Rejected / Low Confidence: {safe_count}/{total_ood} ({rejection_rate*100:.2f}%)")
    print(f"    - unsupported_crop status:     {unsupp_count}/{total_ood} ({unsupp_count/total_ood*100:.2f}%)")
    print(f"    - low_confidence status:       {low_conf_count}/{total_ood} ({low_conf_count/total_ood*100:.2f}%)")
    print(f"    - not_a_leaf status:           {not_leaf_count}/{total_ood} ({not_leaf_count/total_ood*100:.2f}%)")
    print(f"    - False-Positive Diagnoses:    {leak_count}/{total_ood} ({leak_count/total_ood*100:.2f}%)")
    
    print("\n  Rejection by Category:")
    for cat, grp in df_ood_res.groupby("category"):
        c_safe = grp["is_safely_flagged"].sum()
        c_tot = len(grp)
        print(f"    {cat:<14}: {c_safe:>2}/{c_tot:>2} ({c_safe/c_tot*100:5.1f}%) safely rejected")

    # 5. PIL Pipeline Benchmark for Comparison
    print("\n--- [C] PIL Pipeline Evaluation (Reference) ---")
    lab_res_pil = run_eval(session, config_data, lab_df, get_pil_tensor, name="Lab (PIL)")
    field_res_pil = run_eval(session, config_data, field_df, get_pil_tensor, name="Field (PIL)")
    print(f"  Lab Test (PIL):   Accuracy = {lab_res_pil['accuracy']*100:.2f}%, Macro F1 = {lab_res_pil['macro_f1']*100:.2f}%")
    print(f"  Field Test (PIL): Accuracy = {field_res_pil['accuracy']*100:.2f}%, Macro F1 = {field_res_pil['macro_f1']*100:.2f}%")

    # 6. Acceptance Check
    # Criterion: Lab acc drop <= 1.0 point, Field acc does not drop
    v110_lab_alb = 0.9569
    v110_field_alb = 0.4675
    lab_drop = v110_lab_alb - lab_res_alb["accuracy"]
    field_drop = v110_field_alb - field_res_alb["accuracy"]
    
    lab_pass = (lab_drop <= 0.0100)
    field_pass = (field_drop <= 0.0000)
    overall_pass = lab_pass and field_pass
    
    print("\n" + "=" * 70)
    print("ACCEPTANCE VERDICT FOR MODEL v1.1.1:")
    print(f"  Lab Acc Drop:   {lab_drop*100:+.2f}% pts (Threshold: <= +1.00% pt) -> {'PASS' if lab_pass else 'FAIL'}")
    print(f"  Field Acc Drop: {field_drop*100:+.2f}% pts (Threshold: <= 0.00% pt)  -> {'PASS' if field_pass else 'FAIL'}")
    print(f"  OVERALL VERDICT: {'ACCEPTED' if overall_pass else 'REJECTED'}")
    print("=" * 70)

    # Save summary json
    summary = {
        "model_version": "1.1.1",
        "temperature": config_data.get("temperature"),
        "confidence_threshold": config_data.get("confidence_threshold"),
        "albumentations_pipeline": {
            "lab_accuracy": lab_res_alb["accuracy"],
            "lab_macro_f1": lab_res_alb["macro_f1"],
            "lab_coverage": lab_res_alb["coverage"],
            "lab_selective_accuracy": lab_res_alb["selective_accuracy"],
            "field_accuracy": field_res_alb["accuracy"],
            "field_macro_f1": field_res_alb["macro_f1"],
            "field_coverage": field_res_alb["coverage"],
            "field_selective_accuracy": field_res_alb["selective_accuracy"]
        },
        "pil_pipeline": {
            "lab_accuracy": lab_res_pil["accuracy"],
            "lab_macro_f1": lab_res_pil["macro_f1"],
            "field_accuracy": field_res_pil["accuracy"],
            "field_macro_f1": field_res_pil["macro_f1"]
        },
        "false_healthy_rates": {
            "overall": overall_false_healthy,
            "potato": potato_false_healthy,
            "rice": rice_false_healthy
        },
        "novel_ood_benchmark": {
            "total_images": total_ood,
            "safely_rejected": safe_count,
            "rejection_rate": rejection_rate,
            "unsupported_crop_count": unsupp_count,
            "low_confidence_count": low_conf_count,
            "false_positive_count": leak_count
        },
        "acceptance": {
            "lab_accuracy_drop": float(lab_drop),
            "field_accuracy_drop": float(field_drop),
            "lab_pass": bool(lab_pass),
            "field_pass": bool(field_pass),
            "overall_accepted": bool(overall_pass)
        }
    }
    
    with open("scratch/v111_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[+] Saved scratch/v111_evaluation_summary.json")

if __name__ == "__main__":
    main()
