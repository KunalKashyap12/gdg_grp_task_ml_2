"""
Evaluates real test photos from real_test/ on:
- Model v1.0.0 (models/crop_disease_model.onnx)
- Model v1.1.0 (models_v1.1.0/crop_disease_model.onnx)
- Model v1.1.1 (models_v1.1.1/crop_disease_model.onnx, once available)

Reports:
- Predicted class / label
- Confidence
- Returned status (success, low_confidence, not_a_leaf, unsupported_crop)
- Per-image summary table
"""

import os
import sys
import json
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
from src.serve.inference import CropDiseaseInferenceEngine

def evaluate_folder_with_model(model_dir, image_files):
    os.environ["MODEL_DIR"] = model_dir
    onnx_path = os.path.join(model_dir, "crop_disease_model.onnx")
    config_path = os.path.join(model_dir, "model_config.json")
    
    if not os.path.exists(onnx_path) or not os.path.exists(config_path):
        return None
        
    engine = CropDiseaseInferenceEngine(
        onnx_model_path=onnx_path,
        config_path=config_path
    )
    
    results = []
    for f in image_files:
        p = os.path.join("real_test", f)
        with open(p, "rb") as fp:
            img_bytes = fp.read()
            
        res = engine.predict_v1(img_bytes, crop_hint=None, top_k=3)
        status = res.get("status")
        pred = res.get("prediction")
        label = pred.get("label") if pred else "None"
        conf = pred.get("confidence", 0.0) if pred else 0.0
        crop = pred.get("crop") if pred else "None"
        disease = pred.get("disease") if pred else "None"
        
        results.append({
            "filename": f,
            "status": status,
            "label": label,
            "crop": crop,
            "disease": disease,
            "confidence": conf
        })
    return results

def main():
    folder = "data/real_test" if os.path.exists("data/real_test") else "real_test"
    if not os.path.exists(folder):
        print(f"[!] {folder} folder does not exist.")
        return
        
    image_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"[*] Found {len(image_files)} images in {folder}/")
    
    # 1. Evaluate v1.0.0
    v100_dir = "models/archive/v1.0.0" if os.path.exists("models/archive/v1.0.0") else "models_backup_v1.0.0"
    if os.path.exists(v100_dir):
        print("\n" + "=" * 75)
        print(f"Evaluating Model v1.0.0 ({v100_dir}) on {folder}/")
        print("=" * 75)
        res_v100 = evaluate_folder_with_model(v100_dir, image_files)
        df_v100 = pd.DataFrame(res_v100)
        for _, r in df_v100.iterrows():
            print(f"  {r['filename']:<30} | {r['status']:<16} | Conf: {r['confidence']*100:5.1f}% | Label: {r['label']}")
    else:
        res_v100 = None

    # 2. Evaluate v1.1.0
    v110_dir = "models/archive/v1.1.0" if os.path.exists("models/archive/v1.1.0") else "models_v1.1.0"
    if os.path.exists(v110_dir):
        print("\n" + "=" * 75)
        print(f"Evaluating Model v1.1.0 ({v110_dir}) on {folder}/")
        print("=" * 75)
        res_v110 = evaluate_folder_with_model(v110_dir, image_files)
        df_v110 = pd.DataFrame(res_v110)
        for _, r in df_v110.iterrows():
            print(f"  {r['filename']:<30} | {r['status']:<16} | Conf: {r['confidence']*100:5.1f}% | Label: {r['label']}")
    else:
        res_v110 = None

    # 3. Evaluate v1.1.1 (active in models/)
    v111_dir = "models" if os.path.exists("models/crop_disease_model.onnx") else "models/archive/v1.1.1"
    print("\n" + "=" * 75)
    print(f"Evaluating Active Model v1.1.1 ({v111_dir}) on {folder}/")
    print("=" * 75)
    res_v111 = evaluate_folder_with_model(v111_dir, image_files)
    df_v111 = pd.DataFrame(res_v111)
    for _, r in df_v111.iterrows():
        print(f"  {r['filename']:<30} | {r['status']:<16} | Conf: {r['confidence']*100:5.1f}% | Label: {r['label']}")
        
    # Save comparison dataframe
    comp_rows = []
    for idx, f in enumerate(image_files):
        row = {"filename": f}
        if res_v100:
            row["v100_status"] = res_v100[idx]["status"]
            row["v100_label"] = res_v100[idx]["label"]
            row["v100_conf"] = round(res_v100[idx]["confidence"], 4)
        if res_v110:
            row["v110_status"] = res_v110[idx]["status"]
            row["v110_label"] = res_v110[idx]["label"]
            row["v110_conf"] = round(res_v110[idx]["confidence"], 4)
        if res_v111:
            row["v111_status"] = res_v111[idx]["status"]
            row["v111_label"] = res_v111[idx]["label"]
            row["v111_conf"] = round(res_v111[idx]["confidence"], 4)
        comp_rows.append(row)
        
    df_comp = pd.DataFrame(comp_rows)
    os.makedirs("reports", exist_ok=True)
    df_comp.to_csv("reports/real_test_comparison.csv", index=False)
    print("\n[+] Saved per-image comparison table to reports/real_test_comparison.csv")

if __name__ == "__main__":
    main()
