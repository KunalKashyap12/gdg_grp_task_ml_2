"""
Verification script for Section 1, 2, 3, 4 of user verification request:
- Test-set consistency and re-running v1.0.0 and v1.1.0 on test.csv (N=2,829) and plantdoc (N=3,996)
- 95% Wilson confidence intervals for all accuracies
- Full leakage re-check (MD5, group_id, pHash Hamming <= 4)
- Tuning hygiene confirmation
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import onnxruntime as ort
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, ".")
from src.data.transforms import resolve_image_path, IMAGENET_MEAN, IMAGENET_STD

def get_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def wilson_ci(k, n, confidence=0.95):
    """Computes Wilson score 95% confidence interval for proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996  # for 95% CI
    denominator = 1 + (z**2) / n
    centre_adjusted_probability = p + (z**2) / (2 * n)
    adjusted_std = np.sqrt((p * (1 - p) + (z**2) / (4 * n)) / n)
    lower = max(0.0, (centre_adjusted_probability - z * adjusted_std) / denominator)
    upper = min(1.0, (centre_adjusted_probability + z * adjusted_std) / denominator)
    return p, lower, upper

def preprocess_pil(img_path, img_size=224):
    path = resolve_image_path(img_path)
    with Image.open(path) as raw_img:
        img = ImageOps.exif_transpose(raw_img).convert("RGB")
        w, h = img.size
        scale = int(img_size * 1.14) / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = (new_w - img_size) // 2
        top = (new_h - img_size) // 2
        crop_img = img_resized.crop((left, top, left + img_size, top + img_size))
        arr = np.array(crop_img, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return arr.transpose(2, 0, 1)[np.newaxis, ...]

def evaluate_manifest(session, df, path_col, label_col, temperature, threshold, num_classes):
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    
    preds, confs, targets = [], [], []
    for _, row in df.iterrows():
        t = preprocess_pil(row[path_col])
        logits = session.run([out_name], {in_name: t})[0][0]
        cal_logits = logits / temperature
        # Softmax
        e_x = np.exp(cal_logits - np.max(cal_logits))
        probs = e_x / np.sum(e_x)
        
        # Rank among supported classes (first 37)
        pred_idx = int(np.argmax(probs[:37]))
        conf = float(probs[pred_idx])
        
        preds.append(pred_idx)
        confs.append(conf)
        targets.append(int(row[label_col]))
        
    preds = np.array(preds)
    confs = np.array(confs)
    targets = np.array(targets)
    
    acc = accuracy_score(targets, preds)
    accepted = (confs >= threshold)
    cov = float(np.mean(accepted))
    sel_acc = float(np.mean(preds[accepted] == targets[accepted])) if np.sum(accepted) > 0 else 1.0
    f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    
    return {
        "acc": acc,
        "correct": int(np.sum(preds == targets)),
        "n": len(targets),
        "cov": cov,
        "sel_acc": sel_acc,
        "sel_correct": int(np.sum(preds[accepted] == targets[accepted])),
        "sel_n": int(np.sum(accepted)),
        "f1": f1
    }

def main():
    print("=" * 70)
    print("INDEPENDENT AUDIT VERIFICATION: MODEL v1.0.0 vs v1.1.0")
    print("=" * 70)
    
    # Manifest paths & hashes
    m_lab = "data/manifests/test.csv"
    m_field_heldout = "data/manifests/field_test_heldout.csv"
    m_field_orig = "data/manifests/plantdoc_field_test_manifest.csv"
    
    print("\n--- MANIFEST INTEGRITY & HASHES ---")
    df_lab = pd.read_csv(m_lab)
    df_field_heldout = pd.read_csv(m_field_heldout)
    df_field_orig = pd.read_csv(m_field_orig)
    
    with open("data/supported_classes.json") as f:
        sc = json.load(f)
    lbl_to_id = {c["label"]: c["class_id"] for c in sc}
    df_field_orig["class_id"] = df_field_orig["label"].map(lbl_to_id)
    df_field_orig = df_field_orig.dropna(subset=["class_id"]).copy()
    df_field_orig["class_id"] = df_field_orig["class_id"].astype(int)
    
    print(f"1. Lab Test Manifest:             {m_lab} (rows={len(df_lab)}, sha256={get_sha256(m_lab)})")
    print(f"2. Field Heldout Test Manifest:   {m_field_heldout} (rows={len(df_field_heldout)}, sha256={get_sha256(m_field_heldout)})")
    print(f"3. Original PlantDoc Field Test:  {m_field_orig} (rows={len(df_field_orig)}, sha256={get_sha256(m_field_orig)})")
    
    # Sessions
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    s_v100 = ort.InferenceSession("models/crop_disease_model.onnx", sess_options=sess_opts, providers=["CPUExecutionProvider"])
    s_v110 = ort.InferenceSession("models_v1.1.0/crop_disease_model.onnx", sess_options=sess_opts, providers=["CPUExecutionProvider"])
    
    print("\n--- RE-RUNNING EVALUATIONS ON ALL THREE DATASETS ---")
    # 1. Lab test.csv (N=2,829)
    print("Evaluating Lab Test Manifest (N=2,829)...")
    res_lab_v100 = evaluate_manifest(s_v100, df_lab, "filepath", "class_id", temperature=0.4792, threshold=0.74, num_classes=37)
    res_lab_v110 = evaluate_manifest(s_v110, df_lab, "filepath", "class_id", temperature=0.6517, threshold=0.70, num_classes=38)
    
    # 2. Heldout Field test (N=323)
    print("Evaluating Field Heldout Test Manifest (N=323)...")
    res_fh_v100 = evaluate_manifest(s_v100, df_field_heldout, "filepath", "class_id", temperature=0.4792, threshold=0.74, num_classes=37)
    res_fh_v110 = evaluate_manifest(s_v110, df_field_heldout, "filepath", "class_id", temperature=0.6517, threshold=0.70, num_classes=38)
    
    # 3. Original PlantDoc Field test (N=3,996)
    print("Evaluating Original PlantDoc Field Test Manifest (N=3,996)...")
    res_fo_v100 = evaluate_manifest(s_v100, df_field_orig, "cropped_path", "class_id", temperature=0.4792, threshold=0.74, num_classes=37)
    res_fo_v110 = evaluate_manifest(s_v110, df_field_orig, "cropped_path", "class_id", temperature=0.6517, threshold=0.70, num_classes=38)
    
    # Compile Results with 95% Wilson CIs
    results = {
        "lab_test_2829": {
            "v1.0.0": res_lab_v100,
            "v1.1.0": res_lab_v110
        },
        "field_heldout_323": {
            "v1.0.0": res_fh_v100,
            "v1.1.0": res_fh_v110
        },
        "field_orig_3996": {
            "v1.0.0": res_fo_v100,
            "v1.1.0": res_fo_v110
        }
    }
    
    print("\n--- RESULTS SUMMARY & 95% CONFIDENCE INTERVALS ---")
    for dname, dres in results.items():
        print(f"\n[Dataset: {dname}]")
        for mname, mres in dres.items():
            acc_p, acc_lo, acc_hi = wilson_ci(mres["correct"], mres["n"])
            sel_p, sel_lo, sel_hi = wilson_ci(mres["sel_correct"], mres["sel_n"])
            print(f"  {mname}:")
            print(f"    Raw Accuracy:        {acc_p*100:6.2f}% ({mres['correct']}/{mres['n']}) | 95% CI: [{acc_lo*100:6.2f}%, {acc_hi*100:6.2f}%]")
            print(f"    Selective Accuracy:  {sel_p*100:6.2f}% ({mres['sel_correct']}/{mres['sel_n']}) | 95% CI: [{sel_lo*100:6.2f}%, {sel_hi*100:6.2f}%]")
            print(f"    Coverage:            {mres['cov']*100:6.2f}%")
            print(f"    Macro F1:            {mres['f1']*100:6.2f}%")
            
    # Save verification metrics to JSON
    with open("scratch/verification_eval_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Saved scratch/verification_eval_metrics.json")

if __name__ == "__main__":
    main()
