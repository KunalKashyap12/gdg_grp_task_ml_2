"""
Downloads ~500 PlantVillage Orange images and prepares v1.1.1 manifests with zero leakage.
"""

import os
import sys
import json
import hashlib
import urllib.request
import concurrent.futures
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

def compute_md5(fpath):
    with open(fpath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def compute_hashes(img):
    import imagehash
    p_h = str(imagehash.phash(img))
    d_h = str(imagehash.dhash(img))
    return p_h, d_h

def download_img(item, out_dir):
    fname = item["name"]
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return out_path, fname
    req = urllib.request.Request(item["download_url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path, fname

def main():
    print("=" * 70)
    print("TASK 3: PREPARING PLANTVILLAGE ORANGE DATA FOR v1.1.1")
    print("=" * 70)
    
    out_dir = "data/processed_v1.1.1/orange_reject"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Fetch listing of Orange images from spMohanty/PlantVillage-Dataset
    api_url = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/Orange___Haunglongbing_(Citrus_greening)"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        items = json.loads(r.read())
        
    print(f"Total Orange items available on GitHub: {len(items)}")
    # Cap to exactly 500 images
    items_to_download = items[:500]
    print(f"Downloading capped sample of {len(items_to_download)} images...")
    
    downloaded = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_img, it, out_dir): it for it in items_to_download}
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                downloaded.append(res)
            except Exception as e:
                pass
                
    print(f"Successfully downloaded {len(downloaded)} Orange images to {out_dir}")
    
    # 2. Process images, compute hashes and group_id
    records = []
    for fpath, fname in downloaded:
        try:
            with Image.open(fpath) as img:
                img_rgb = ImageOps.exif_transpose(img).convert("RGB")
                w, h = img_rgb.size
                p_h, d_h = compute_hashes(img_rgb)
            # Group ID: In PlantVillage, the prefix before '___' is the unique plant leaf sample ID
            stem = os.path.splitext(fname)[0]
            group_id = stem.split("___")[0] if "___" in stem else stem
            
            records.append({
                "filepath": fpath.replace("\\", "/"),
                "filename": fname,
                "class_label": "Unsupported___Unsupported_crop",
                "class_id": 37,
                "crop": "Unsupported",
                "disease": "Unsupported_crop",
                "group_id": f"pv_orange_{group_id}",
                "md5": compute_md5(fpath),
                "phash": p_h,
                "dhash": d_h,
                "width": w,
                "height": h,
                "split": ""
            })
        except Exception as e:
            print(f"Failed processing {fname}: {e}")
            
    df_orange = pd.DataFrame(records)
    print(f"Processed DataFrame: {len(df_orange)} records across {df_orange['group_id'].nunique()} unique groups")
    
    # 3. Split by Group: 80% Train, 20% Val
    unique_groups = df_orange["group_id"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    split_idx = int(0.80 * len(unique_groups))
    train_groups = set(unique_groups[:split_idx])
    val_groups = set(unique_groups[split_idx:])
    
    df_orange_train = df_orange[df_orange["group_id"].isin(train_groups)].copy()
    df_orange_train["split"] = "train"
    df_orange_val = df_orange[df_orange["group_id"].isin(val_groups)].copy()
    df_orange_val["split"] = "val"
    
    print(f"Orange Split: {len(df_orange_train)} Train images ({len(train_groups)} groups) | {len(df_orange_val)} Val images ({len(val_groups)} groups)")
    
    # 4. Merge into v1.1.1 Manifests
    df_train_v110 = pd.read_csv("data/manifests/train_v1.1.0.csv")
    df_val_v110 = pd.read_csv("data/manifests/val_v1.1.0.csv")
    
    # Combine
    df_train_v111 = pd.concat([df_train_v110, df_orange_train], ignore_index=True)
    df_val_v111 = pd.concat([df_val_v110, df_orange_val], ignore_index=True)
    
    m_train_v111 = "data/manifests/train_v1.1.1.csv"
    m_val_v111 = "data/manifests/val_v1.1.1.csv"
    df_train_v111.to_csv(m_train_v111, index=False)
    df_val_v111.to_csv(m_val_v111, index=False)
    
    print(f"[+] Saved {m_train_v111}: {len(df_train_v111)} samples (was {len(df_train_v110)})")
    print(f"[+] Saved {m_val_v111}: {len(df_val_v111)} samples (was {len(df_val_v110)})")
    
    # 5. Zero-Leakage Verification against Test Sets
    df_test_lab = pd.read_csv("data/manifests/test.csv")
    df_test_field = pd.read_csv("data/manifests/field_test_heldout.csv")
    
    train_md5 = set(df_train_v111["md5"])
    lab_md5 = set(df_test_lab["md5"])
    field_md5 = set(df_test_field["md5"])
    
    print("\n--- LEAKAGE VERIFICATION FOR v1.1.1 ---")
    print("MD5 collisions Train <-> Lab Test:   ", len(train_md5 & lab_md5))
    print("MD5 collisions Train <-> Field Test: ", len(train_md5 & field_md5))
    assert len(train_md5 & lab_md5) == 0, "Train vs Lab Test MD5 leak!"
    assert len(train_md5 & field_md5) == 0, "Train vs Field Test MD5 leak!"
    
    train_grp = set(df_train_v111["group_id"])
    lab_grp = set(df_test_lab["group_id"])
    field_grp = set(df_test_field["group_id"])
    print("group_id collisions Train <-> Lab Test:   ", len(train_grp & lab_grp))
    print("group_id collisions Train <-> Field Test: ", len(train_grp & field_grp))
    assert len(train_grp & lab_grp) == 0, "Train vs Lab Test group_id leak!"
    assert len(train_grp & field_grp) == 0, "Train vs Field Test group_id leak!"
    
    print("[+] ZERO DATA LEAKAGE CONFIRMED FOR v1.1.1 DATASETS!")

if __name__ == "__main__":
    main()
