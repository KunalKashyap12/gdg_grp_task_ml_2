"""
Crop-Hint Evaluation Script:
Evaluates v1.1.0 on lab test set (test.csv) and field test set (field_test_heldout.csv):
1. Correct crop hint at thresholds 0.35, 0.50, 0.70
2. Wrong crop hint at thresholds 0.35, 0.50, 0.70
3. Proposed safe threshold & disagreement gating mechanism
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import onnxruntime as ort
from sklearn.metrics import accuracy_score

sys.path.insert(0, ".")
from src.data.transforms import get_val_transforms, resolve_image_path

def main():
    print("=" * 70)
    print("TASK 2: CROP-HINT MODE BENCHMARK & THRESHOLD ANALYSIS")
    print("=" * 70)
    
    transform = get_val_transforms(img_size=224)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession("models_v1.1.0/crop_disease_model.onnx", sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    
    with open("data/supported_classes.json") as f:
        classes_list = json.load(f)
    class_map = {c["class_id"]: c for c in classes_list}
    crops = sorted(list(set(c["crop"] for c in classes_list)))
    crop_to_indices = {crop: [c["class_id"] for c in classes_list if c["crop"] == crop] for crop in crops}
    
    T = 0.6517
    
    def run_eval(df_path, name):
        df = pd.read_csv(df_path)
        print(f"\nEvaluating {name} ({len(df)} samples)...")
        
        # Precompute logits
        all_logits = []
        all_targets = []
        all_crops = []
        
        for _, r in df.iterrows():
            p = resolve_image_path(r["filepath"])
            with Image.open(p) as img:
                img_np = np.array(ImageOps.exif_transpose(img).convert("RGB"))
            t = transform(image=img_np)["image"].numpy()[np.newaxis, ...]
            raw_logits = session.run([out_name], {in_name: t})[0][0]
            all_logits.append(raw_logits)
            all_targets.append(int(r["class_id"]))
            all_crops.append(r["crop"])
            
        all_logits = np.array(all_logits)
        all_targets = np.array(all_targets)
        
        # Global softmax & crop species probabilities
        cal_logits = all_logits / T
        e_x = np.exp(cal_logits - np.max(cal_logits, axis=1, keepdims=True))
        cal_probs = e_x / np.sum(e_x, axis=1, keepdims=True)
        
        # Crop marginals
        crop_probs = np.zeros((len(df), len(crops)))
        for c_idx, crop in enumerate(crops):
            crop_indices = crop_to_indices[crop]
            crop_probs[:, c_idx] = np.sum(cal_probs[:, crop_indices], axis=1)
            
        top_crop_indices = np.argmax(crop_probs, axis=1)
        top_crops = [crops[i] for i in top_crop_indices]
        top_crop_confs = np.max(crop_probs, axis=1)
        
        thresholds = [0.35, 0.50, 0.70]
        results = {}
        
        # 1. Baseline: Unconditioned
        uncond_preds = np.argmax(cal_probs[:, :37], axis=1)
        uncond_confs = np.max(cal_probs[:, :37], axis=1)
        print("\n--- UNCONDITIONED (NO HINT) ---")
        for th in thresholds:
            acc_mask = (uncond_confs >= th)
            cov = float(np.mean(acc_mask))
            raw_acc = float(np.mean(uncond_preds == all_targets))
            sel_acc = float(np.mean(uncond_preds[acc_mask] == all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 1.0
            wrong_diag = float(np.mean(uncond_preds[acc_mask] != all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 0.0
            print(f"  Threshold {th:0.2f} | Cov: {cov*100:5.2f}% | Sel Acc: {sel_acc*100:5.2f}% | Wrong Diag: {wrong_diag*100:5.2f}%")
            results[f"unconditioned_th_{th}"] = {"cov": cov, "sel_acc": sel_acc, "wrong_diag": wrong_diag}
            
        # 2. CORRECT CROP HINT
        print("\n--- CORRECT CROP HINT ---")
        correct_preds = []
        correct_confs = []
        for i in range(len(df)):
            c_name = all_crops[i]
            c_inds = crop_to_indices[c_name]
            sub_l = all_logits[i, c_inds] / T
            sub_p = np.exp(sub_l - np.max(sub_l)) / np.sum(np.exp(sub_l - np.max(sub_l)))
            best_rank = np.argmax(sub_p)
            correct_preds.append(c_inds[best_rank])
            correct_confs.append(float(sub_p[best_rank]))
            
        correct_preds = np.array(correct_preds)
        correct_confs = np.array(correct_confs)
        
        for th in thresholds:
            acc_mask = (correct_confs >= th)
            cov = float(np.mean(acc_mask))
            sel_acc = float(np.mean(correct_preds[acc_mask] == all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 1.0
            wrong_diag = float(np.mean(correct_preds[acc_mask] != all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 0.0
            print(f"  Threshold {th:0.2f} | Cov: {cov*100:5.2f}% | Sel Acc: {sel_acc*100:5.2f}% | Wrong Diag: {wrong_diag*100:5.2f}%")
            results[f"correct_hint_th_{th}"] = {"cov": cov, "sel_acc": sel_acc, "wrong_diag": wrong_diag}
            
        # 3. WRONG CROP HINT (Shifted crop cyclically)
        print("\n--- WRONG CROP HINT (e.g. Tomato with hint 'Apple') ---")
        wrong_preds = []
        wrong_confs = []
        hint_crop_probs = []
        for i in range(len(df)):
            true_crop = all_crops[i]
            # Shift to next crop in alphabetical order
            true_c_idx = crops.index(true_crop)
            wrong_crop = crops[(true_c_idx + 1) % len(crops)]
            
            c_inds = crop_to_indices[wrong_crop]
            sub_l = all_logits[i, c_inds] / T
            sub_p = np.exp(sub_l - np.max(sub_l)) / np.sum(np.exp(sub_l - np.max(sub_l)))
            best_rank = np.argmax(sub_p)
            wrong_preds.append(c_inds[best_rank])
            wrong_confs.append(float(sub_p[best_rank]))
            
            # Probability model assigned to the wrong crop
            hint_crop_idx = crops.index(wrong_crop)
            hint_crop_probs.append(crop_probs[i, hint_crop_idx])
            
        wrong_preds = np.array(wrong_preds)
        wrong_confs = np.array(wrong_confs)
        hint_crop_probs = np.array(hint_crop_probs)
        
        for th in thresholds:
            # Under naive crop conditioning: acceptance only checks sub_conf >= th
            acc_mask = (wrong_confs >= th)
            cov = float(np.mean(acc_mask))
            # True accuracy is 0% since crop is fundamentally wrong
            sel_acc = float(np.mean(wrong_preds[acc_mask] == all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 0.0
            wrong_diag = float(np.mean(wrong_preds[acc_mask] != all_targets[acc_mask])) if np.sum(acc_mask) > 0 else 0.0
            print(f"  Naive Threshold {th:0.2f} | Accepted/Cov: {cov*100:5.2f}% | Wrong Diag Rate: {wrong_diag*100:5.2f}%")
            results[f"wrong_hint_naive_th_{th}"] = {"cov": cov, "sel_acc": sel_acc, "wrong_diag": wrong_diag}
            
        # 4. PROPOSED SAFE CROP HINT DISAGREEMENT GUARD
        # Guard rule:
        # Require:
        # a) sub_conf >= 0.70 (standard threshold)
        # b) Disagreement check:
        #    If model's unconditioned top crop prediction != hint crop,
        #    and the hint crop probability < 0.20 (or top crop confidence > 0.60),
        #    reject as low_confidence!
        print("\n--- PROPOSED SAFE CROP HINT WITH DISAGREEMENT GATING ---")
        # On Correct Hint
        safe_correct_acc = []
        for i in range(len(df)):
            c_name = all_crops[i]
            hint_p = crop_probs[i, crops.index(c_name)]
            sub_c = correct_confs[i]
            disagree = (top_crops[i] != c_name) and (hint_p < 0.20 or top_crop_confs[i] > 0.70)
            accepted = (sub_c >= 0.70) and not disagree
            safe_correct_acc.append(accepted)
        safe_correct_acc = np.array(safe_correct_acc)
        cov_corr = float(np.mean(safe_correct_acc))
        sel_acc_corr = float(np.mean(correct_preds[safe_correct_acc] == all_targets[safe_correct_acc]))
        print(f"  Correct Hint with Safe Guard: Coverage: {cov_corr*100:5.2f}%, Selective Acc: {sel_acc_corr*100:5.2f}%")
        
        # On Wrong Hint
        safe_wrong_acc = []
        for i in range(len(df)):
            true_crop = all_crops[i]
            wrong_crop = crops[(crops.index(true_crop) + 1) % len(crops)]
            hint_p = crop_probs[i, crops.index(wrong_crop)]
            sub_c = wrong_confs[i]
            disagree = (top_crops[i] != wrong_crop) and (hint_p < 0.20 or top_crop_confs[i] > 0.70)
            accepted = (sub_c >= 0.70) and not disagree
            safe_wrong_acc.append(accepted)
        safe_wrong_acc = np.array(safe_wrong_acc)
        cov_wrong = float(np.mean(safe_wrong_acc))
        print(f"  Wrong Hint with Safe Guard:   Accepted/Leakage Rate: {cov_wrong*100:5.2f}% (Dropped from {results['wrong_hint_naive_th_0.35']['cov']*100:.1f}%)")
        
        return results, cov_corr, sel_acc_corr, cov_wrong

    res_lab, c_cov_l, c_acc_l, w_cov_l = run_eval("data/manifests/test.csv", "Lab Test Set")
    res_fld, c_cov_f, c_acc_f, w_cov_f = run_eval("data/manifests/field_test_heldout.csv", "Field Held-Out Test Set")
    
    with open("scratch/crop_hint_metrics.json", "w") as f:
        json.dump({
            "lab": res_lab,
            "field": res_fld,
            "safe_guard": {
                "lab_correct_cov": c_cov_l,
                "lab_correct_sel_acc": c_acc_l,
                "lab_wrong_leakage": w_cov_l,
                "field_correct_cov": c_cov_f,
                "field_correct_sel_acc": c_acc_f,
                "field_wrong_leakage": w_cov_f
            }
        }, f, indent=2)
    print("\n[+] Saved scratch/crop_hint_metrics.json")

if __name__ == "__main__":
    main()
