"""
Dataset Preparation, Cleaning, Duplicate Detection, and Partitioning Pipeline.
- Enforces 10 crops and 37 classes defined in config.yaml
- Uses PIL and ImageHash (pHash) to filter corrupted images and remove near-duplicates
- Balances class distributions up to max_images_per_class
- Verifies every class has at least 100 images (or flags low-reliability)
- Creates stratified train (70%), val (15%), test (15%) splits
- Exports manifests and organized PyTorch ImageFolder hierarchy
"""

import os
import glob
import shutil
import json
import yaml
import random
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


# Direct folder name to target standard label mapping
FOLDER_MAPPINGS = {
    # Apple
    "Apple___Apple_scab": "Apple___Apple_scab",
    "Apple___Black_rot": "Apple___Black_rot",
    "Apple___Cedar_apple_rust": "Apple___Cedar_apple_rust",
    "Apple___healthy": "Apple___Healthy",
    # Cherry
    "Cherry_(including_sour)___Powdery_mildew": "Cherry___Powdery_mildew",
    "Cherry_(including_sour)___healthy": "Cherry___Healthy",
    # Maize
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Maize___Gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "Maize___Common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Maize___Northern_leaf_blight",
    "Corn_(maize)___healthy": "Maize___Healthy",
    # Grape
    "Grape___Black_rot": "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)": "Grape___Esca_black_measles",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape___Leaf_blight",
    "Grape___healthy": "Grape___Healthy",
    # Peach
    "Peach___Bacterial_spot": "Peach___Bacterial_spot",
    "Peach___healthy": "Peach___Healthy",
    # Pepper
    "Pepper,_bell___Bacterial_spot": "Pepper___Bacterial_spot",
    "Pepper,_bell___healthy": "Pepper___Healthy",
    # Potato
    "Potato___Early_blight": "Potato___Early_blight",
    "Potato___Late_blight": "Potato___Late_blight",
    "Potato___healthy": "Potato___Healthy",
    # Strawberry
    "Strawberry___Leaf_scorch": "Strawberry___Leaf_scorch",
    "Strawberry___healthy": "Strawberry___Healthy",
    # Tomato
    "Tomato___Bacterial_spot": "Tomato___Bacterial_spot",
    "Tomato___Early_blight": "Tomato___Early_blight",
    "Tomato___Late_blight": "Tomato___Late_blight",
    "Tomato___Leaf_Mold": "Tomato___Leaf_mold",
    "Tomato___Septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato___Spider_mites",
    "Tomato___Target_Spot": "Tomato___Target_spot",
    "Tomato___Tomato_mosaic_virus": "Tomato___Mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato___Yellow_leaf_curl_virus",
    "Tomato___healthy": "Tomato___Healthy",
    # Rice (from test_rice2)
    "bacterial_leaf_blight": "Rice___Bacterial_leaf_blight",
    "brown_spot": "Rice___Brown_spot",
    "rice_blast": "Rice___Leaf_blast",
    "healthy_rice_plant": "Rice___Healthy",
}


def find_source_images():
    """Locate all raw images from PlantVillage download and Rice dataset."""
    found_by_target = {}

    # 1. PlantVillage from download_pv
    pv_base = "download_pv/PlantVillage"
    if os.path.exists(pv_base):
        for split in ["train", "val"]:
            s_dir = os.path.join(pv_base, split)
            if not os.path.exists(s_dir):
                continue
            for folder_name in os.listdir(s_dir):
                if folder_name in FOLDER_MAPPINGS:
                    target_label = FOLDER_MAPPINGS[folder_name]
                    folder_path = os.path.join(s_dir, folder_name)
                    images = glob.glob(os.path.join(folder_path, "*.*"))
                    found_by_target.setdefault(target_label, []).extend(
                        [("PlantVillage", "lab", p) for p in images]
                    )

    # 2. Rice from test_rice2
    rice_base = "test_rice2"
    if os.path.exists(rice_base):
        for root, dirs, files in os.walk(rice_base):
            folder_name = os.path.basename(root)
            if folder_name in FOLDER_MAPPINGS:
                target_label = FOLDER_MAPPINGS[folder_name]
                images = [os.path.join(root, f) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                found_by_target.setdefault(target_label, []).extend(
                    [("Philippines_Rice", "field", p) for p in images]
                )

    return found_by_target


def clean_and_deduplicate(images_info, max_images=350, phash_threshold=4):
    """
    Validates images and removes exact or near duplicates via perceptual hash.
    Returns cleaned list of (source, img_type, path).
    """
    hashes = []
    cleaned = []
    duplicates_removed = 0
    corrupt_removed = 0

    # Shuffle to avoid order bias before capping
    random.shuffle(images_info)

    for source, img_type, img_path in images_info:
        if len(cleaned) >= max_images:
            break
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                h = imagehash.phash(img_rgb)
        except Exception:
            corrupt_removed += 1
            continue

        # Check for near-duplicates
        is_dup = False
        for existing_h in hashes:
            if h - existing_h <= phash_threshold:
                is_dup = True
                break

        if is_dup:
            duplicates_removed += 1
        else:
            hashes.append(h)
            cleaned.append((source, img_type, img_path))

    return cleaned, duplicates_removed, corrupt_removed


def process_dataset(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["system"].get("seed", 42)
    set_seed(seed)

    with open("data/supported_classes.json", "r", encoding="utf-8") as f:
        supported_classes = json.load(f)
    valid_labels = set(c["label"] for c in supported_classes)

    max_per_class = cfg["data"].get("max_images_per_class", 350)
    min_per_class = cfg["data"].get("min_images_per_class", 100)
    phash_th = cfg["data"].get("phash_threshold", 4)
    processed_dir = cfg["data"]["processed_dir"]
    manifest_dir = cfg["data"]["manifest_dir"]

    print(f"[*] Discovering raw source images across all 37 classes...")
    source_dict = find_source_images()

    manifest_records = []
    class_stats = []

    print(f"[*] Cleaning, validating, and deduplicating images (max {max_per_class}/class)...")

    for entry in supported_classes:
        label = entry["label"]
        crop = entry["crop"]
        disease = entry["disease"]
        raw_list = source_dict.get(label, [])
        raw_count = len(raw_list)

        cleaned, dups, corrupt = clean_and_deduplicate(raw_list, max_images=max_per_class, phash_threshold=phash_th)
        clean_count = len(cleaned)

        status = "reliable" if clean_count >= min_per_class else "low-reliability"
        if clean_count == 0:
            status = "missing"

        class_stats.append({
            "class_id": entry["class_id"],
            "label": label,
            "crop": crop,
            "disease": disease,
            "raw_count": raw_count,
            "clean_count": clean_count,
            "duplicates_removed": dups,
            "corrupt_removed": corrupt,
            "status": status
        })

        for source, img_type, p in cleaned:
            manifest_records.append({
                "class_id": entry["class_id"],
                "label": label,
                "crop": crop,
                "disease": disease,
                "source": source,
                "photo_type": img_type,
                "raw_path": os.path.abspath(p)
            })

    df = pd.DataFrame(manifest_records)
    print(f"[+] Total cleaned dataset size: {len(df)} images across {df['label'].nunique()} classes")

    # Stratified Train / Val / Test Partitioning
    train_pct = 0.70
    val_pct = 0.15
    test_pct = 0.15

    train_rows, val_rows, test_rows = [], [], []

    for label, group in df.groupby("label"):
        n = len(group)
        indices = list(range(n))
        random.shuffle(indices)

        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_rows.append(group.iloc[train_idx])
        val_rows.append(group.iloc[val_idx])
        test_rows.append(group.iloc[test_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    master_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Save manifests
    os.makedirs(manifest_dir, exist_ok=True)
    master_df.to_csv(os.path.join(manifest_dir, "manifest.csv"), index=False)
    train_df.to_csv(os.path.join(manifest_dir, "train_manifest.csv"), index=False)
    val_df.to_csv(os.path.join(manifest_dir, "val_manifest.csv"), index=False)
    test_df.to_csv(os.path.join(manifest_dir, "test_manifest.csv"), index=False)

    print(f"[+] Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Copy files into data/processed/ for standard PyTorch ImageFolder access
    print("[*] Materializing processed ImageFolder directory structure...")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = os.path.join(processed_dir, split_name)
        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Writing {split_name}"):
            dest_dir = os.path.join(split_dir, row["label"])
            os.makedirs(dest_dir, exist_ok=True)
            fname = os.path.basename(row["raw_path"])
            dest_path = os.path.join(dest_dir, fname)
            if not os.path.exists(dest_path):
                shutil.copy2(row["raw_path"], dest_path)

    # Save stats summary
    stats_df = pd.DataFrame(class_stats)
    stats_path = "reports/phase1_dataset_audit.csv"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"[+] Dataset audit saved to {stats_path}")

    return stats_df, master_df


if __name__ == "__main__":
    process_dataset()
