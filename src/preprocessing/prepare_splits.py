
import argparse
import os
import json
import shutil
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

HAM_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']
NAME_TO_IDX = {n:i for i,n in enumerate(HAM_CLASSES)}

def _safe_symlink_or_copy(src, dst, copy=False):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        if copy:
            shutil.copy2(src, dst)
        else:
            # attempt symlink, fallback to copy on failure (e.g., Windows without privileges)
            os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def _gather_images(raw_dir):
    # Kaggle structure: HAM10000_images_part_1/*.jpg and _part_2/*.jpg
    parts = sorted(glob(os.path.join(raw_dir, "HAM10000_images_part_*", "*.jpg")))
    # Some mirrors keep images in a single folder; include that too
    single = sorted(glob(os.path.join(raw_dir, "*.jpg")))
    images = parts + single
    if not images:
        raise FileNotFoundError("No .jpg images found under {}".format(raw_dir))
    return images

def _load_metadata(raw_dir):
    meta_path = os.path.join(raw_dir, "HAM10000_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing HAM10000_metadata.csv in {raw_dir}")
    df = pd.read_csv(meta_path)
    # Standardize column names if needed
    # Columns include: lesion_id, image_id, dx, dx_type, age, sex, localization
    return df

def _compute_mean_std(image_paths, sample_size=5000, image_size=224):
    from torchvision import transforms
    import torch
    tfm = transforms.Compose([
        transforms.Resize(int(1.15*image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    if len(image_paths) > sample_size:
        rng = np.random.RandomState(42)
        image_paths = rng.choice(image_paths, size=sample_size, replace=False)
    n = 0
    mean = torch.zeros(3)
    M2 = torch.zeros(3)
    for p in tqdm(image_paths, desc="computing mean/std"):
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            continue
        x = tfm(img).view(3, -1)  # 3 x HW
        x_mean = x.mean(dim=1)
        x_var = x.var(dim=1, unbiased=False)
        n += 1
        delta = x_mean - mean
        mean += delta / n
        delta2 = x_mean - mean
        M2 += x_var + delta * delta2
    var = M2 / n
    std = var.sqrt()
    return mean.tolist(), std.tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--img_size", default=224, type=int)
    ap.add_argument("--test_size", default=0.15, type=float)
    ap.add_argument("--val_size", default=0.15, type=float)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--copy_images", default=0, type=int, help="1=copy files, 0=symlink if possible")
    ap.add_argument("--compute_stats", default=1, type=int)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_out_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_out_dir, exist_ok=True)
    splits_dir = os.path.join(args.out_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    meta = _load_metadata(args.raw_dir)
    images = _gather_images(args.raw_dir)
    # Map image_id to path
    id_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in images}

    # Build a master table
    rows = []
    missing = 0
    for _, r in meta.iterrows():
        image_id = r["image_id"]
        src = id_to_path.get(image_id, None)
        if src is None:
            missing += 1
            continue
        dst = os.path.join(img_out_dir, os.path.basename(src))
        _safe_symlink_or_copy(src, dst, copy=bool(args.copy_images))
        label_name = str(r["dx"]).strip()
        if label_name not in NAME_TO_IDX:
            # Occasionally some dumps use 'vasc' vs 'vasc ' etc.
            label_name = label_name.replace(" ", "")
        if label_name not in NAME_TO_IDX:
            raise ValueError(f"Unknown class label in metadata: {label_name}")
        rows.append({
            "image_id": image_id,
            "image_path": dst,
            "label_name": label_name,
            "label": NAME_TO_IDX[label_name],
            "lesion_id": r.get("lesion_id", ""),
            "age": r.get("age", ""),
            "sex": r.get("sex", ""),
            "localization": r.get("localization", ""),
        })
    if missing:
        print(f"Warning: {missing} metadata entries with missing image files")

    all_df = pd.DataFrame(rows)
    all_csv = os.path.join(splits_dir, "all.csv")
    all_df.to_csv(all_csv, index=False)

    # Stratified split on label
    trainval_df, test_df = train_test_split(
        all_df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=all_df["label"],
    )
    # Further split train/val
    val_rel = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_rel,
        random_state=args.seed,
        stratify=trainval_df["label"],
    )

    train_csv = os.path.join(splits_dir, "train.csv"); train_df.to_csv(train_csv, index=False)
    val_csv   = os.path.join(splits_dir, "val.csv");   val_df.to_csv(val_csv, index=False)
    test_csv  = os.path.join(splits_dir, "test.csv");  test_df.to_csv(test_csv, index=False)

    # Save label map
    label_map = {c:i for i,c in enumerate(HAM_CLASSES)}
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    # Optional: compute mean/std (on a sample for speed)
    if args.compute_stats:
        mean, std = _compute_mean_std(list(train_df["image_path"]), image_size=args.img_size)
        with open(os.path.join(args.out_dir, "stats.json"), "w") as f:
            json.dump({"mean": mean, "std": std, "image_size": args.img_size}, f, indent=2)

    print("Done. Splits saved to", splits_dir)

if __name__ == "__main__":
    main()
