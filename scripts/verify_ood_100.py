"""
Downloads 105 test images of plants & objects NOT used in the reject class:
- 50 images of Orange (PlantVillage: Citrus Greening / HLB) - an out-of-scope crop
- 30 images of other plants/weeds/garden plants
- 25 images of non-plant random objects
Then runs InferenceEngine on each to measure the rejection / low-confidence rate.
"""

import os
import sys
import json
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
from src.serve.inference import CropDiseaseInferenceEngine

def download_file(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)

def main():
    print("=" * 70)
    print("OOD TEST: 100+ IMAGES NOT USED IN THE REJECT CLASS")
    print("=" * 70)
    
    out_dir = "scratch/ood_100_images"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Fetch 50 Orange images from PlantVillage via GitHub API
    print("\n[1] Fetching 50 Orange leaf images from PlantVillage...")
    api_url = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/Orange___Haunglongbing_(Citrus_greening)"
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        items = json.loads(r.read())
    
    orange_items = items[:50]
    orange_paths = []
    for idx, item in enumerate(orange_items):
        fname = f"orange_{idx:03d}.jpg"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            try:
                download_file(item["download_url"], fpath)
            except Exception as e:
                print(f"Failed {item['name']}: {e}")
        if os.path.exists(fpath):
            orange_paths.append((fpath, "Orange (Citrus Greening)", "other_crop"))
            
    print(f"Successfully obtained {len(orange_paths)} Orange crop images.")
    
    # 2. Curate 30 garden plants, weeds, wildflowers
    # Use reliable Wikimedia / public URLs
    print("\n[2] Fetching 30 garden plants, weeds, and wildflowers...")
    plant_urls = [
        # Dandelion, Clover, Fern, Rose, Sunflower, Grass, Pine, Bamboo, Moss, Ivy, Orchid, Mint, Basil
        ("https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?w=400", "Pine needles", "garden_plant"),
        ("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=400", "Fern leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1463936575829-25148e1db1b8?w=400", "Cactus", "garden_plant"),
        ("https://images.unsplash.com/photo-1507290439931-a861b5a38200?w=400", "Dandelion weed", "weed"),
        ("https://images.unsplash.com/photo-1528183429752-a97d0bf99b5a?w=400", "Wild grass", "weed"),
        ("https://images.unsplash.com/photo-1546842931-886c185b4c8c?w=400", "Sunflower petal/leaf", "garden_plant"),
        ("https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=400", "Wildflower blossom", "garden_plant"),
        ("https://images.unsplash.com/photo-1533038590840-1cde6e668a91?w=400", "Monstera houseplant", "houseplant"),
        ("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?w=400", "Green foliage garden", "garden_plant"),
        ("https://images.unsplash.com/photo-1470058869958-2a77ade41c02?w=400", "Lush jungle leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400", "Autumn maple leaf", "tree_leaf"),
        ("https://images.unsplash.com/photo-1448375240586-882707db888b?w=400", "Forest moss and fern", "weed"),
        ("https://images.unsplash.com/photo-1512428813834-c702c7702b78?w=400", "Succulent plant", "houseplant"),
        ("https://images.unsplash.com/photo-1485955900006-10f4d324d411?w=400", "Potted aloe plant", "houseplant"),
        ("https://images.unsplash.com/photo-1497250681960-ef046c08a56e?w=400", "Palm frond", "garden_plant"),
        ("https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400", "Tree bark and ivy", "weed"),
        ("https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400", "Lawn clover patch", "weed"),
        ("https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=400", "Sunlit forest foliage", "garden_plant"),
        ("https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400", "Bamboo leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400", "Banana leaf", "other_crop"),
        ("https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?w=400", "Orchid flower and leaf", "houseplant"),
        ("https://images.unsplash.com/photo-1508873696983-2df5293cb325?w=400", "Tea bush foliage", "other_crop"),
        ("https://images.unsplash.com/photo-1471193945509-9ad0617afabf?w=400", "Rose shrub with leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=400", "Alpine moss", "weed"),
        ("https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=400", "Oak tree leaf canopy", "tree_leaf"),
        ("https://images.unsplash.com/photo-1516205651411-aef33a44f7c2?w=400", "Tulip garden leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=400", "Lotus pad in water", "garden_plant"),
        ("https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?w=400", "Wheat field blades", "other_crop"),
        ("https://images.unsplash.com/photo-1508746829417-e6f548d8d6ed?w=400", "Fiddle leaf fig", "houseplant"),
        ("https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400", "Mint / herb leaves", "garden_plant")
    ]
    
    plant_paths = []
    for idx, (url, desc, cat) in enumerate(plant_urls):
        fname = f"plant_{idx:03d}.jpg"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            try:
                download_file(url, fpath)
            except Exception as e:
                print(f"Failed plant {idx}: {e}")
        if os.path.exists(fpath):
            plant_paths.append((fpath, desc, cat))
    print(f"Successfully obtained {len(plant_paths)} plant/weed images.")
    
    # 3. Curate 25 non-plant random objects
    print("\n[3] Fetching 25 non-plant random objects...")
    object_urls = [
        ("https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400", "Smartphone", "object"),
        ("https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400", "Wristwatch", "object"),
        ("https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400", "Headphones", "object"),
        ("https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400", "Red sneaker shoe", "object"),
        ("https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=400", "Camera lens", "object"),
        ("https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400", "Sunglasses", "object"),
        ("https://images.unsplash.com/photo-1560343090-f0409e92791a?w=400", "Leather shoes", "object"),
        ("https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400", "Earphones", "object"),
        ("https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400", "Smart watch", "object"),
        ("https://images.unsplash.com/photo-1585386959984-a4155224a1ad?w=400", "Perfume bottle", "object"),
        ("https://images.unsplash.com/photo-1524805444758-089113d48a6d?w=400", "Classic watch", "object"),
        ("https://images.unsplash.com/photo-1503602642458-232111445657?w=400", "Wooden chair", "object"),
        ("https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "Apple MacBook laptop", "object"),
        ("https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400", "Computer mouse", "object"),
        ("https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=400", "Mechanical keyboard", "object"),
        ("https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=400", "Golden retriever dog", "animal"),
        ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400", "Tabby cat", "animal"),
        ("https://images.unsplash.com/photo-1552053831-71594a27632d?w=400", "Dog portrait", "animal"),
        ("https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400", "Beagle dog face", "animal"),
        ("https://images.unsplash.com/photo-1533738363-b7f9aef128ce?w=400", "Cat with glasses", "animal"),
        ("https://images.unsplash.com/photo-1579783902614-a3fb3927b675?w=400", "Oil painting abstract art", "texture"),
        ("https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=400", "Wooden table texture", "texture"),
        ("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400", "Geometric wallpaper", "texture"),
        ("https://images.unsplash.com/photo-1508615039623-a25605d2b022?w=400", "Blue abstract wallpaper", "texture"),
        ("https://images.unsplash.com/photo-1557683316-973673baf926?w=400", "Gradient color wallpaper", "texture")
    ]
    
    object_paths = []
    for idx, (url, desc, cat) in enumerate(object_urls):
        fname = f"object_{idx:03d}.jpg"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            try:
                download_file(url, fpath)
            except Exception as e:
                print(f"Failed object {idx}: {e}")
        if os.path.exists(fpath):
            object_paths.append((fpath, desc, cat))
    print(f"Successfully obtained {len(object_paths)} object/wallpaper images.")
    
    all_eval_items = orange_paths + plant_paths + object_paths
    total_images = len(all_eval_items)
    print(f"\n[+] Total out-of-scope images assembled for evaluation: {total_images}")
    
    # 4. Evaluate using production InferenceEngine (v1.1.0)
    print("\n[*] Initializing CropDiseaseInferenceEngine (Model v1.1.0)...")
    os.environ["MODEL_DIR"] = "models_v1.1.0"
    engine = CropDiseaseInferenceEngine(
        onnx_model_path="models_v1.1.0/crop_disease_model.onnx",
        config_path="models_v1.1.0/model_config.json"
    )
    
    results = []
    for fpath, desc, cat in all_eval_items:
        with open(fpath, "rb") as f:
            img_bytes = f.read()
        res = engine.predict_v1(img_bytes, crop_hint=None, top_k=3)
        status = res.get("status")
        conf = res.get("prediction", {}).get("confidence", 0.0) if res.get("prediction") else 0.0
        label = res.get("prediction", {}).get("label") if res.get("prediction") else None
        
        # Flagged as unsupported or low_confidence or not_a_leaf
        is_safe = (status in ["unsupported_crop", "low_confidence", "not_a_leaf"])
        
        results.append({
            "path": fpath,
            "description": desc,
            "category": cat,
            "status": status,
            "confidence": conf,
            "predicted_label": label,
            "is_safely_flagged": is_safe
        })
        
    df_res = pd.DataFrame(results)
    
    # Breakdown
    total = len(df_res)
    num_unsupported = int(np.sum(df_res["status"] == "unsupported_crop"))
    num_low_conf = int(np.sum(df_res["status"] == "low_confidence"))
    num_not_leaf = int(np.sum(df_res["status"] == "not_a_leaf"))
    num_success = int(np.sum(df_res["status"] == "success"))
    
    total_safe = num_unsupported + num_low_conf + num_not_leaf
    safe_fraction = total_safe / total
    
    print("\n" + "=" * 50)
    print(f"OUT-OF-SCOPE EVALUATION SUMMARY (N={total} images)")
    print("=" * 50)
    print(f"1. Flagged as 'unsupported_crop': {num_unsupported:3d} ({num_unsupported/total*100:5.1f}%)")
    print(f"2. Flagged as 'low_confidence':   {num_low_conf:3d} ({num_low_conf/total*100:5.1f}%)")
    print(f"3. Flagged as 'not_a_leaf':      {num_not_leaf:3d} ({num_not_leaf/total*100:5.1f}%)")
    print(f"4. False Positive 'success':      {num_success:3d} ({num_success/total*100:5.1f}%)")
    print("-" * 50)
    print(f"TOTAL SAFELY FLAGGED:             {total_safe:3d} / {total} ({safe_fraction*100:5.1f}%)")
    
    print("\n--- BREAKDOWN BY CATEGORY ---")
    for cat in df_res["category"].unique():
        sub = df_res[df_res["category"] == cat]
        safe_sub = np.sum(sub["is_safely_flagged"])
        print(f"  Category '{cat:<14}' (N={len(sub):2d}): {safe_sub:2d}/{len(sub):2d} safely flagged ({safe_sub/len(sub)*100:5.1f}%) | "
              f"unsupported={np.sum(sub['status']=='unsupported_crop')}, low_conf={np.sum(sub['status']=='low_confidence')}, not_leaf={np.sum(sub['status']=='not_a_leaf')}")
        
    # Save results to CSV & JSON
    df_res.to_csv("scratch/ood_100_results.csv", index=False)
    with open("scratch/ood_100_metrics.json", "w") as f:
        json.dump({
            "total_tested": total,
            "safely_flagged_count": total_safe,
            "safely_flagged_fraction": safe_fraction,
            "unsupported_crop_count": num_unsupported,
            "low_confidence_count": num_low_conf,
            "not_a_leaf_count": num_not_leaf,
            "false_positive_success_count": num_success
        }, f, indent=2)
    print("\n[+] Saved scratch/ood_100_results.csv and scratch/ood_100_metrics.json")

if __name__ == "__main__":
    main()
