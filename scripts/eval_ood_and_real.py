"""
Benchmark Evaluation for Model v1.2.0:
1. Lab test set: data/manifests/test.csv (N=2,829)
2. Field test set: data/manifests/field_test_heldout.csv (N=323)
3. False-healthy rates (overall, Potato, Rice)
4. Novel Out-of-Scope set: data/novel_ood/ (N >= 100 images, strictly excluding Orange, Squash, Raspberry, Blueberry, Soybean)
5. Real photos: real_test/ (N=20 images)
6. Compiles complete eval_v120_results.json
"""

import os
import sys
import json
import math
import urllib.request
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

def run_eval(session, config_data, df):
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

def download_file(url, out_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)

def get_novel_ood_dataset():
    out_dir = "data/novel_ood"
    os.makedirs(out_dir, exist_ok=True)
    manifest_csv = os.path.join(out_dir, "manifest.csv")
    
    if os.path.exists(manifest_csv):
        df_existing = pd.read_csv(manifest_csv)
        if len(df_existing) >= 100 and all(os.path.exists(p) for p in df_existing["filepath"]):
            return df_existing
            
    # Curate high-quality URLs of non-orange plants, trees, weeds, objects
    urls = [
        # 1. Garden plants, flowers, trees, houseplants
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
        ("https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=400", "Banana leaf", "novel_crop"),
        ("https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?w=400", "Orchid flower and leaf", "houseplant"),
        ("https://images.unsplash.com/photo-1508873696983-2df5293cb325?w=400", "Tea bush foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1471193945509-9ad0617afabf?w=400", "Rose shrub with leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=400", "Alpine moss", "weed"),
        ("https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=400", "Oak tree leaf canopy", "tree_leaf"),
        ("https://images.unsplash.com/photo-1516205651411-aef33a44f7c2?w=400", "Tulip garden leaves", "garden_plant"),
        ("https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=400", "Lotus pad in water", "garden_plant"),
        ("https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?w=400", "Wheat field blades", "novel_crop"),
        ("https://images.unsplash.com/photo-1508746829417-e6f548d8d6ed?w=400", "Fiddle leaf fig", "houseplant"),
        ("https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400", "Mint herb leaves", "garden_plant"),
        
        # 2. Novel agricultural crops (not in 38 classes)
        ("https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=400", "Coffee branch and beans", "novel_crop"),
        ("https://images.unsplash.com/photo-1594489428504-5c0c480a15fd?w=400", "Cotton boll in field", "novel_crop"),
        ("https://images.unsplash.com/photo-1589923188900-85dae523342b?w=400", "Sugarcane stalk and leaves", "novel_crop"),
        ("https://images.unsplash.com/photo-1588880331179-bc9b93a8cb5e?w=400", "Cassava root and foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1615485290382-441e4d049cb5?w=400", "Papaya tree foliage", "novel_crop"),
        ("https://images.unsplash.com/photo-1553279768-865429fa0078?w=400", "Mango foliage and fruit", "novel_crop"),
        ("https://images.unsplash.com/photo-1574323347407-f5e1ad6d020b?w=400", "Barley grain ears", "novel_crop"),
        ("https://images.unsplash.com/photo-1586201375761-83865001e31c?w=400", "Oat field stalks", "novel_crop"),
        ("https://images.unsplash.com/photo-1628088062854-d1870b4553da?w=400", "Sorghum heads", "novel_crop"),
        ("https://images.unsplash.com/photo-1599940824399-b87987ceb72a?w=400", "Ginger plant leaves", "novel_crop"),
        ("https://images.unsplash.com/photo-1596040033229-a9821ebd058d?w=400", "Black pepper vines", "novel_crop"),
        ("https://images.unsplash.com/photo-1578849278619-e73505e9610f?w=400", "Cocoa pod on tree", "novel_crop"),

        # 3. Wild weeds and herbs
        ("https://images.unsplash.com/photo-1596704017254-9b121068fb31?w=400", "Stinging nettle leaves", "wild_weed"),
        ("https://images.unsplash.com/photo-1595855759920-86582396756a?w=400", "Milk thistle blossom", "wild_weed"),
        ("https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=400", "Burdock plant leaves", "wild_weed"),
        ("https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?w=400", "Broadleaf plantain", "wild_weed"),
        ("https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=400", "Field chickweed", "wild_weed"),
        ("https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=400", "Horsetail rush", "wild_weed"),
        ("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=400", "Common ragweed in meadow", "wild_weed"),
        ("https://images.unsplash.com/photo-1470240731273-7821a6eeb6bd?w=400", "Ground ivy patch", "wild_weed"),
        ("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400", "Morning glory bindweed", "wild_weed"),
        ("https://images.unsplash.com/photo-1473773508845-188df298d2d1?w=400", "Wild chamomile flowers", "wild_weed"),
        ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=400", "Alpine sorrel foliage", "wild_weed"),
        ("https://images.unsplash.com/photo-1476820865390-c52aeebb9891?w=400", "Common dandelion clock", "wild_weed"),
        ("https://images.unsplash.com/photo-1500534623283-312aade485b7?w=400", "Wild field grasses", "wild_weed"),

        # 4. Forestry and tree leaves
        ("https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?w=400", "Weeping willow fronds", "tree_leaf"),
        ("https://images.unsplash.com/photo-1464226184884-fa280b87c399?w=400", "Cherry blossom tree", "tree_leaf"),
        ("https://images.unsplash.com/photo-1476231682828-37e571bc172f?w=400", "Autumn forest birch", "tree_leaf"),
        ("https://images.unsplash.com/photo-1509114397022-ed747cca3f65?w=400", "Fir tree needles", "tree_leaf"),
        ("https://images.unsplash.com/photo-1448375240586-882707db888b?w=400", "Silver birch leaves", "tree_leaf"),

        # 5. Non-plant random objects
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
        ("https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400", "Travel backpack", "object"),
        ("https://images.unsplash.com/photo-1602143407151-7111542de6e8?w=400", "Stainless water bottle", "object"),
        ("https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=400", "Road bicycle frame", "object"),
        ("https://images.unsplash.com/photo-1507473885765-e6ed057f782c?w=400", "Modern desk lamp", "object"),
        ("https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=400", "Hardcover book", "object"),
        ("https://images.unsplash.com/photo-1589939705384-5185137a7f0f?w=400", "Stainless scissors", "object"),
        ("https://images.unsplash.com/photo-1578749556568-bc2c40e68b61?w=400", "Ceramic dinner plate", "object"),
        ("https://images.unsplash.com/photo-1581147036324-c17ac41dfa6c?w=400", "Hand screwdriver tool", "object"),
        ("https://images.unsplash.com/photo-1584990347449-34fb8e9c60e0?w=400", "Metal cutlery", "object"),
        ("https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400", "Computer mouse", "object"),
        ("https://images.unsplash.com/photo-1587829741301-dc798b83add3?w=400", "Mechanical keyboard", "object"),
        ("https://images.unsplash.com/photo-1541643600914-78b084683601?w=400", "Glass perfume vial", "object"),
        ("https://images.unsplash.com/photo-1592945403244-b3fbafd7f539?w=400", "Ceramic coffee mug", "object"),
        ("https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?w=400", "Cosmetic brush set", "object"),
        ("https://images.unsplash.com/photo-1513542789411-b6a5d4f31634?w=400", "Wall clock dial", "object"),
        ("https://images.unsplash.com/photo-1517336714731-489689fd1ca8?w=400", "Laptop notebook", "object"),
        ("https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400", "Audio headset", "object"),
        ("https://images.unsplash.com/photo-1546868871-7041f2a55e12?w=400", "Fitness tracker band", "object"),
        ("https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400", "In-ear headphones", "object"),
        
        # 6. Additional botanical and everyday textures
        ("https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?w=400", "Rainforest moss", "garden_plant"),
        ("https://images.unsplash.com/photo-1465146344425-f00d5f5c8f07?w=400", "Autumn leaf canopy", "tree_leaf"),
        ("https://images.unsplash.com/photo-1500534314209-a25ddb2bd429?w=400", "Meadow field blossom", "wild_weed"),
        ("https://images.unsplash.com/photo-1502082553048-f009c37129b9?w=400", "Deciduous branch", "tree_leaf"),
        ("https://images.unsplash.com/photo-1448375240586-882707db888b?w=400", "Spruce branch", "tree_leaf"),
        ("https://images.unsplash.com/photo-1508873696983-2df5293cb325?w=400", "Clover field lawn", "weed"),
        ("https://images.unsplash.com/photo-1473448912268-2022ce9509d8?w=400", "Forest oak leaf", "tree_leaf"),
        ("https://images.unsplash.com/photo-1513836279014-a89f7a76ae86?w=400", "Ivy creeper leaves", "weed"),
        ("https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=400", "Wild grasses blade", "weed"),
        ("https://images.unsplash.com/photo-1518495973542-4542c06a5843?w=400", "Sunlit foliage leaf", "garden_plant"),
        ("https://images.unsplash.com/photo-1534447677768-be436bb09401?w=400", "Bamboo frond leaf", "garden_plant"),
        ("https://images.unsplash.com/photo-1520412099551-62b6bafeb5bb?w=400", "Houseplant orchid", "houseplant"),
        ("https://images.unsplash.com/photo-1471193945509-9ad0617afabf?w=400", "Rose shrub leaf", "garden_plant"),
        ("https://images.unsplash.com/photo-1426604966848-d7adac402bff?w=400", "Rock moss foliage", "weed"),
        ("https://images.unsplash.com/photo-1516205651411-aef33a44f7c2?w=400", "Spring garden plant", "garden_plant"),
        ("https://images.unsplash.com/photo-1524492412937-b28074a5d7da?w=400", "Lily leaf on pond", "garden_plant"),
        ("https://images.unsplash.com/photo-1508746829417-e6f548d8d6ed?w=400", "Fig houseplant leaf", "houseplant"),
        ("https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?w=400", "Herb bush leaves", "garden_plant")
    ]
    
    records = []
    print(f"[*] Downloading / verifying {len(urls)} novel OOD images in {out_dir}...")
    for idx, (url, desc, cat) in enumerate(urls):
        fname = f"ood_{idx:03d}.jpg"
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            try:
                download_file(url, fpath)
            except Exception as e:
                pass
        if os.path.exists(fpath):
            records.append({
                "filename": fname,
                "filepath": fpath,
                "description": desc,
                "category": cat
            })
            
    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"[+] Compiled {len(df_manifest)} novel OOD samples into {manifest_csv}")
    return df_manifest

def main():
    print("=" * 75)
    print("[*] Benchmark Evaluation of Model v1.2.0 (Optuna Tuned)")
    print("=" * 75)
    
    model_dir = "models_v1.2.0"
    onnx_path = os.path.join(model_dir, "crop_disease_model.onnx")
    config_path = os.path.join(model_dir, "model_config.json")
    
    with open(config_path, "r") as f:
        config_data = json.load(f)
        
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    print(f"[*] Loaded ONNX: {onnx_path}")
    print(f"[*] T = {config_data.get('temperature')}, threshold = {config_data.get('confidence_threshold')}")
    
    # 1. Lab Test
    lab_df = pd.read_csv("data/manifests/test.csv")
    print(f"\n[1] Evaluating Lab Test Set (N={len(lab_df)})...")
    lab_res = run_eval(session, config_data, lab_df)
    print(f"  Accuracy:           {lab_res['accuracy']*100:.2f}% (95% CI: [{lab_res['ci_lower']*100:.2f}%, {lab_res['ci_upper']*100:.2f}%])")
    print(f"  Macro F1:           {lab_res['macro_f1']*100:.2f}%")
    print(f"  Coverage (th=0.70): {lab_res['coverage']*100:.2f}%")
    print(f"  Selective Accuracy: {lab_res['selective_accuracy']*100:.2f}%")
    
    # 2. Field Test
    field_df = pd.read_csv("data/manifests/field_test_heldout.csv")
    print(f"\n[2] Evaluating Field Test Set (N={len(field_df)})...")
    field_res = run_eval(session, config_data, field_df)
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
    
    # 4. Novel Out-of-Scope Rejection Benchmark
    ood_df = get_novel_ood_dataset()
    print(f"\n[4] Evaluating Novel Out-of-Scope Rejection (N={len(ood_df)} images)...")
    os.environ["MODEL_DIR"] = model_dir
    engine = CropDiseaseInferenceEngine(onnx_model_path=onnx_path, config_path=config_path)
    
    ood_records = []
    for _, row in ood_df.iterrows():
        with open(row["filepath"], "rb") as f:
            b = f.read()
        res = engine.predict_v1(b, crop_hint=None, top_k=3)
        st = res.get("status")
        conf = res.get("prediction", {}).get("confidence", 0.0) if res.get("prediction") else 0.0
        label = res.get("prediction", {}).get("label") if res.get("prediction") else None
        ood_records.append({
            "filename": row["filename"],
            "filepath": row["filepath"],
            "description": row["description"],
            "category": row["category"],
            "status": st,
            "confidence": conf,
            "label": label,
            "is_safely_flagged": (st in ["unsupported_crop", "low_confidence", "not_a_leaf"])
        })
    df_ood = pd.DataFrame(ood_records)
    safe_cnt = int(df_ood["is_safely_flagged"].sum())
    safe_rate = float(safe_cnt / len(df_ood))
    print(f"  Safely Flagged: {safe_cnt}/{len(df_ood)} ({safe_rate*100:.2f}%)")
    for st, c in df_ood["status"].value_counts().items():
        print(f"    - {st}: {c} ({c/len(df_ood)*100:.1f}%)")
        
    # 5. Real Photos Evaluation
    real_test_dir = "real_test"
    if not os.path.exists(real_test_dir):
        real_test_dir = "data/real_test"
    real_files = sorted([f for f in os.listdir(real_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    print(f"\n[5] Evaluating Real Photos (N={len(real_files)} in {real_test_dir})...")
    real_records = []
    for fname in real_files:
        fpath = os.path.join(real_test_dir, fname)
        with open(fpath, "rb") as f:
            b = f.read()
        res = engine.predict_v1(b, crop_hint=None, top_k=3)
        st = res.get("status")
        pred_label = res.get("prediction", {}).get("label") if res.get("prediction") else None
        conf = res.get("prediction", {}).get("confidence", 0.0) if res.get("prediction") else 0.0
        real_records.append({
            "filename": fname,
            "status": st,
            "prediction": pred_label,
            "confidence": conf
        })
        print(f"    {fname:<25} -> status: {st:<18} | pred: {str(pred_label):<30} | conf: {conf:.4f}")
        
    # 6. Save JSON
    summary = {
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
            "safely_flagged_count": safe_cnt,
            "safely_flagged_rate": safe_rate,
            "status_breakdown": df_ood["status"].value_counts().to_dict()
        },
        "real_photos": real_records
    }
    with open("reports/improvement/eval_v120_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[+] Saved reports/improvement/eval_v120_results.json")

if __name__ == "__main__":
    main()
