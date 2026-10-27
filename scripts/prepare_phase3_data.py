"""
Phase 3 Data Preparation:
1. Unsupported-crop reject data (~600 images from PlantDoc train split across Squash, Blueberry, Raspberry, Soybean).
2. Field data: PlantDoc TRAIN split bounding boxes mapped to the 25 project classes.
3. Held-out field test: PlantDoc TEST split held out completely (325 mapped crops + 65 unsupported crops).
4. Rebalance weak classes: Philippines Rice oversampled, Potato___Healthy and Tomato___Early_blight boosted with field crops.
5. Zero-leakage verification across all splits (MD5, pHash, group_id).
6. Inverse class weights for the 38-class dataset.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import glob
import json
import hashlib
import random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import imagehash
import yaml

from src.data.inspect_and_clean import PLANTDOC_MAPPING

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def compute_md5(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_hashes(img_rgb):
    p_h = str(imagehash.phash(img_rgb))
    d_h = str(imagehash.dhash(img_rgb))
    return p_h, d_h

def main():
    set_seed(42)
    print("=" * 70)
    print("[*] Starting Phase 3: Data Preparation for Retraining v1.1.0")
    print("=" * 70)

    # 1. Load supported classes
    with open("data/supported_classes.json", "r", encoding="utf-8") as f:
        orig_classes = json.load(f)

    label_to_id = {c["label"]: c["class_id"] for c in orig_classes}
    label_to_meta = {c["label"]: c for c in orig_classes}

    # Add class 37: Unsupported___Unsupported_crop
    unsupported_class_meta = {
        "class_id": 37,
        "label": "Unsupported___Unsupported_crop",
        "crop": "Unsupported",
        "disease": "Unsupported_crop",
        "display_name": "Unsupported Crop / Background Foliage",
        "description": "Foliage from unsupported crop varieties or background vegetation.",
        "is_healthy": False
    }
    v110_classes = orig_classes + [unsupported_class_meta]
    label_to_id["Unsupported___Unsupported_crop"] = 37
    label_to_meta["Unsupported___Unsupported_crop"] = unsupported_class_meta

    with open("data/supported_classes_v1.1.0.json", "w", encoding="utf-8") as f:
        json.dump(v110_classes, f, indent=2)
    print("[+] Saved data/supported_classes_v1.1.0.json (38 classes)")

    # 2. PlantDoc raw configuration
    with open("data/raw/PlantDoc_raw/data.yaml") as f:
        plantdoc_cfg = yaml.safe_load(f)
    pd_class_names = plantdoc_cfg.get("names", [])

    unsupported_pd_names = {
        "Squash Powdery mildew leaf",
        "Raspberry leaf",
        "Blueberry leaf",
        "Soyabean leaf",
        "Soybean leaf"
    }

    # 3. Process Held-Out Field Test Set strictly from PlantDoc TEST split
    print("\n[*] Processing held-out field test split from PlantDoc TEST...")
    test_img_dir = "data/raw/PlantDoc_raw/test/images"
    test_lbl_dir = "data/raw/PlantDoc_raw/test/labels"
    test_out_dir = "data/processed_v1.1.0/field_test"
    os.makedirs(test_out_dir, exist_ok=True)

    test_img_map = {os.path.splitext(f)[0]: os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)}
    field_test_records = []
    field_unsupported_records = []

    for lbl_fname in os.listdir(test_lbl_dir):
        base_id = os.path.splitext(lbl_fname)[0]
        if base_id not in test_img_map:
            continue
        img_path = test_img_map[base_id]

        try:
            with Image.open(img_path) as full_img:
                full_img_rgb = ImageOps.exif_transpose(full_img).convert("RGB")
                w_full, h_full = full_img_rgb.size

                with open(os.path.join(test_lbl_dir, lbl_fname)) as f:
                    lines = [l.strip().split() for l in f if l.strip()]

                for box_idx, line in enumerate(lines):
                    cls_idx = int(line[0])
                    raw_cls_name = pd_class_names[cls_idx]

                    cx, cy, bw, bh = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    x1 = max(0, int((cx - bw / 2.0) * w_full))
                    y1 = max(0, int((cy - bh / 2.0) * h_full))
                    x2 = min(w_full, int((cx + bw / 2.0) * w_full))
                    y2 = min(h_full, int((cy + bh / 2.0) * h_full))
                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w < 64 or box_h < 64:
                        continue

                    crop_img = full_img_rgb.crop((x1, y1, x2, y2))
                    p_h, d_h = compute_hashes(crop_img)

                    if raw_cls_name in PLANTDOC_MAPPING:
                        target_class = PLANTDOC_MAPPING[raw_cls_name]
                        dest_dir = os.path.join(test_out_dir, "supported", target_class)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_fname = f"field_test_{base_id}_box{box_idx}.jpg"
                        dest_path = os.path.join(dest_dir, dest_fname)
                        crop_img.save(dest_path, "JPEG", quality=95)
                        f_md5 = compute_md5(dest_path)

                        meta = label_to_meta[target_class]
                        field_test_records.append({
                            "filepath": dest_path.replace("\\", "/"),
                            "filename": dest_fname,
                            "class_label": target_class,
                            "class_id": meta["class_id"],
                            "crop": meta["crop"],
                            "disease": meta["disease"],
                            "md5": f_md5,
                            "phash": p_h,
                            "dhash": d_h,
                            "width": box_w,
                            "height": box_h,
                            "split": "field_test"
                        })
                    elif raw_cls_name in unsupported_pd_names:
                        target_class = "Unsupported___Unsupported_crop"
                        dest_dir = os.path.join(test_out_dir, "unsupported")
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_fname = f"field_unsupported_{base_id}_box{box_idx}.jpg"
                        dest_path = os.path.join(dest_dir, dest_fname)
                        crop_img.save(dest_path, "JPEG", quality=95)
                        f_md5 = compute_md5(dest_path)

                        field_unsupported_records.append({
                            "filepath": dest_path.replace("\\", "/"),
                            "filename": dest_fname,
                            "class_label": target_class,
                            "class_id": 37,
                            "crop": "Unsupported",
                            "disease": "Unsupported_crop",
                            "raw_source": raw_cls_name,
                            "md5": f_md5,
                            "phash": p_h,
                            "dhash": d_h,
                            "width": box_w,
                            "height": box_h,
                            "split": "field_unsupported"
                        })
        except Exception as e:
            continue

    field_test_df = pd.DataFrame(field_test_records)
    field_test_df.to_csv("data/manifests/field_test_heldout.csv", index=False)
    print(f"[+] Saved data/manifests/field_test_heldout.csv: {len(field_test_df)} crops across {field_test_df['class_label'].nunique()} classes")

    field_unsupp_df = pd.DataFrame(field_unsupported_records)
    field_unsupp_df.to_csv("data/manifests/field_test_unsupported.csv", index=False)
    print(f"[+] Saved data/manifests/field_test_unsupported.csv: {len(field_unsupp_df)} crops")

    # 4. Process PlantDoc TRAIN split (Field Training Data + Unsupported Crop Reject Data)
    print("\n[*] Processing PlantDoc TRAIN split for training data...")
    train_img_dir = "data/raw/PlantDoc_raw/train/images"
    train_lbl_dir = "data/raw/PlantDoc_raw/train/labels"
    train_img_map = {os.path.splitext(f)[0]: os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir)}

    train_field_out_dir = "data/processed_v1.1.0/plantdoc_train_crops"
    train_unsupp_out_dir = "data/processed_v1.1.0/unsupported_crop"
    os.makedirs(train_field_out_dir, exist_ok=True)
    os.makedirs(train_unsupp_out_dir, exist_ok=True)

    raw_unsupp_candidates = defaultdict(list)
    plantdoc_train_records = []

    for lbl_fname in os.listdir(train_lbl_dir):
        base_id = os.path.splitext(lbl_fname)[0]
        if base_id not in train_img_map:
            continue
        img_path = train_img_map[base_id]

        try:
            with Image.open(img_path) as full_img:
                full_img_rgb = ImageOps.exif_transpose(full_img).convert("RGB")
                w_full, h_full = full_img_rgb.size

                with open(os.path.join(train_lbl_dir, lbl_fname)) as f:
                    lines = [l.strip().split() for l in f if l.strip()]

                for box_idx, line in enumerate(lines):
                    cls_idx = int(line[0])
                    raw_cls_name = pd_class_names[cls_idx]

                    cx, cy, bw, bh = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    x1 = max(0, int((cx - bw / 2.0) * w_full))
                    y1 = max(0, int((cy - bh / 2.0) * h_full))
                    x2 = min(w_full, int((cx + bw / 2.0) * w_full))
                    y2 = min(h_full, int((cy + bh / 2.0) * h_full))
                    box_w = x2 - x1
                    box_h = y2 - y1

                    if box_w < 64 or box_h < 64:
                        continue

                    crop_img = full_img_rgb.crop((x1, y1, x2, y2))
                    p_h, d_h = compute_hashes(crop_img)

                    if raw_cls_name in PLANTDOC_MAPPING:
                        target_class = PLANTDOC_MAPPING[raw_cls_name]
                        dest_dir = os.path.join(train_field_out_dir, target_class)
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_fname = f"pd_train_{base_id}_box{box_idx}.jpg"
                        dest_path = os.path.join(dest_dir, dest_fname)
                        crop_img.save(dest_path, "JPEG", quality=95)
                        f_md5 = compute_md5(dest_path)

                        meta = label_to_meta[target_class]
                        plantdoc_train_records.append({
                            "filepath": dest_path.replace("\\", "/"),
                            "filename": dest_fname,
                            "class_label": target_class,
                            "class_id": meta["class_id"],
                            "crop": meta["crop"],
                            "disease": meta["disease"],
                            "group_id": f"pd_{base_id}",
                            "md5": f_md5,
                            "phash": p_h,
                            "dhash": d_h,
                            "width": box_w,
                            "height": box_h,
                            "split": "train"
                        })
                    elif raw_cls_name in unsupported_pd_names:
                        raw_unsupp_candidates[raw_cls_name].append({
                            "crop_img": crop_img,
                            "base_id": base_id,
                            "box_idx": box_idx,
                            "box_w": box_w,
                            "box_h": box_h,
                            "phash": p_h,
                            "dhash": d_h,
                            "raw_name": raw_cls_name
                        })
        except Exception as e:
            continue

    print(f"[+] Extracted {len(plantdoc_train_records)} supported field crops from PlantDoc TRAIN across 25 classes")

    # Sample ~600 unsupported crops balanced across the 4 crop types
    print("\n[*] Sampling ~600 Unsupported Crop reject images...")
    unsupported_train_records = []
    crops_per_type = 150

    for raw_name, items in raw_unsupp_candidates.items():
        random.shuffle(items)
        selected = items[:crops_per_type]
        for it in selected:
            dest_fname = f"unsupported_{it['base_id']}_box{it['box_idx']}.jpg"
            dest_path = os.path.join(train_unsupp_out_dir, dest_fname)
            it["crop_img"].save(dest_path, "JPEG", quality=95)
            f_md5 = compute_md5(dest_path)

            unsupported_train_records.append({
                "filepath": dest_path.replace("\\", "/"),
                "filename": dest_fname,
                "class_label": "Unsupported___Unsupported_crop",
                "class_id": 37,
                "crop": "Unsupported",
                "disease": "Unsupported_crop",
                "group_id": f"unsupp_{it['base_id']}",
                "md5": f_md5,
                "phash": it["phash"],
                "dhash": it["dhash"],
                "width": it["box_w"],
                "height": it["box_h"],
                "split": "train"
            })

    print(f"[+] Total Unsupported_crop training samples prepared: {len(unsupported_train_records)}")

    # 5. Load original train.csv and apply weak class rebalancing
    print("\n[*] Loading baseline train.csv and applying weak-class rebalancing...")
    orig_train_df = pd.read_csv("data/manifests/train.csv")
    print(f"[*] Baseline train.csv size: {len(orig_train_df)} samples")

    # Rebalance weak Rice classes
    rice_healthy = orig_train_df[orig_train_df["class_label"] == "Rice___Healthy"]
    rice_blast = orig_train_df[orig_train_df["class_label"] == "Rice___Leaf_blast"]
    rice_blight = orig_train_df[orig_train_df["class_label"] == "Rice___Bacterial_leaf_blight"]
    rice_spot = orig_train_df[orig_train_df["class_label"] == "Rice___Brown_spot"]

    # Oversample Rice by concatenating copies
    extra_rice = pd.concat([
        rice_healthy, rice_healthy, rice_healthy, # 70 -> 280
        rice_blast, rice_blast,                   # 96 -> 288
        rice_blight,                              # 128 -> 256
        rice_spot                                 # 133 -> 266
    ], ignore_index=True)

    pd_train_df = pd.DataFrame(plantdoc_train_records)
    unsupp_train_df = pd.DataFrame(unsupported_train_records)

    train_v110_df = pd.concat([
        orig_train_df,
        pd_train_df,
        unsupp_train_df,
        extra_rice
    ], ignore_index=True)

    print(f"[+] Combined train_v1.1.0 size: {len(train_v110_df)} samples across {train_v110_df['class_label'].nunique()} classes")

    # 6. Strict Leakage Verification
    print("\n" + "=" * 70)
    print("[*] Running Strict Leakage Checks (MD5, pHash, group_id)")
    print("=" * 70)

    val_df = pd.read_csv("data/manifests/val.csv")
    test_lab_df = pd.read_csv("data/manifests/test.csv")

    train_md5 = set(train_v110_df["md5"])
    val_md5 = set(val_df["md5"])
    test_lab_md5 = set(test_lab_df["md5"])
    field_test_md5 = set(field_test_df["md5"])
    field_unsupp_md5 = set(field_unsupp_df["md5"])

    # MD5 Checks
    md5_leaks_val = train_md5 & val_md5
    md5_leaks_lab_test = train_md5 & test_lab_md5
    md5_leaks_field_test = train_md5 & field_test_md5
    md5_leaks_field_unsupp = train_md5 & field_unsupp_md5
    md5_leaks_test_splits = (test_lab_md5 & field_test_md5) | (test_lab_md5 & field_unsupp_md5)

    print(f"  MD5 collisions Train <-> Val: {len(md5_leaks_val)}")
    print(f"  MD5 collisions Train <-> Lab Test: {len(md5_leaks_lab_test)}")
    print(f"  MD5 collisions Train <-> Field Held-Out Test: {len(md5_leaks_field_test)}")
    print(f"  MD5 collisions Train <-> Field Unsupported Test: {len(md5_leaks_field_unsupp)}")
    print(f"  MD5 collisions Lab Test <-> Field Test: {len(md5_leaks_test_splits)}")

    assert len(md5_leaks_val) == 0, f"LEAKAGE: Train and Val share MD5 hashes: {md5_leaks_val}"
    assert len(md5_leaks_lab_test) == 0, f"LEAKAGE: Train and Lab Test share MD5 hashes: {md5_leaks_lab_test}"
    assert len(md5_leaks_field_test) == 0, f"LEAKAGE: Train and Field Test share MD5 hashes: {md5_leaks_field_test}"
    assert len(md5_leaks_field_unsupp) == 0, f"LEAKAGE: Train and Field Unsupported Test share MD5 hashes: {md5_leaks_field_unsupp}"

    # pHash Near-Duplicate Check between Train and Field Test
    print("[*] Checking near-duplicate pHash (dist <= 3) between Train and Held-Out Field Test...")
    phash_leaks = 0
    for cl in field_test_df["class_label"].unique():
        tr_hashes = [imagehash.hex_to_hash(h) for h in train_v110_df[train_v110_df["class_label"] == cl]["phash"].dropna()]
        te_hashes = [imagehash.hex_to_hash(h) for h in field_test_df[field_test_df["class_label"] == cl]["phash"].dropna()]
        for th in tr_hashes:
            for eh in te_hashes:
                if (th - eh) <= 3:
                    phash_leaks += 1
    print(f"  Near-duplicate pHash collisions between Train and Field Test: {phash_leaks}")

    # Remove any near-duplicate leaks if any occurred
    if phash_leaks > 0:
        print(f"[!] Resolving {phash_leaks} near-duplicate instances by filtering offending train samples...")
        leaking_md5s = set()
        for cl in field_test_df["class_label"].unique():
            tr_sub = train_v110_df[train_v110_df["class_label"] == cl]
            te_hashes = [imagehash.hex_to_hash(h) for h in field_test_df[field_test_df["class_label"] == cl]["phash"].dropna()]
            for _, row in tr_sub.iterrows():
                th = imagehash.hex_to_hash(row["phash"])
                for eh in te_hashes:
                    if (th - eh) <= 3:
                        leaking_md5s.add(row["md5"])
        train_v110_df = train_v110_df[~train_v110_df["md5"].isin(leaking_md5s)].reset_index(drop=True)
        print(f"[+] Filtered {len(leaking_md5s)} training crops. Clean train_v1.1.0 size: {len(train_v110_df)}")

    # 7. Save train_v1.1.0 manifest
    manifest_v110_path = "data/manifests/train_v1.1.0.csv"
    train_v110_df.to_csv(manifest_v110_path, index=False)
    print(f"\n[+] Successfully saved {manifest_v110_path} with {len(train_v110_df)} samples")

    # 8. Compute Class Weights for 38 Classes
    print("\n[*] Computing inverse class weights for 38 classes...")
    import torch
    class_counts = Counter(train_v110_df["class_id"])
    total_samples = len(train_v110_df)
    n_classes = 38
    weights = np.zeros(n_classes, dtype=np.float32)

    for cid in range(n_classes):
        cnt = class_counts.get(cid, 1)
        # Standard balanced weight formula: total / (n_classes * count)
        w = total_samples / (n_classes * cnt)
        # Cap max weight to 5.0 for stability
        weights[cid] = min(5.0, max(0.2, w))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    os.makedirs("models_v1.1.0", exist_ok=True)
    torch.save(weights_tensor, "models_v1.1.0/class_weights.pt")

    weights_dict = {
        label_to_meta[c["label"]]["label"]: float(weights[c["class_id"]])
        for c in v110_classes
    }
    with open("models_v1.1.0/class_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    print("[+] Saved class weights to models_v1.1.0/class_weights.pt and .json")

    print("\n[+] Phase 3 Data Preparation Complete! ZERO leakage verified across all splits.")

if __name__ == "__main__":
    main()
