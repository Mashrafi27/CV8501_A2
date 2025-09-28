
import argparse
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.data.ham10000 import Ham10000Dataset
from src.models.baseline_cnn import create_model
from src.evaluation.metrics import compute_metrics, save_metrics_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, type=str)
    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--batch_size", default=128, type=int)
    ap.add_argument("--num_workers", default=4, type=int)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load stats if present
    stats_path = os.path.join(os.path.dirname(os.path.dirname(args.test_csv)), "stats.json")
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    if os.path.exists(stats_path):
        try:
            s = json.load(open(stats_path))
            mean, std = s.get("mean", mean), s.get("std", std)
        except Exception:
            pass

    t_test = build_transforms(args.image_size, train=False, mean=mean, std=std)
    test_ds = Ham10000Dataset(args.test_csv, transform=t_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Load model config from checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt.get("config", {})
    model_name = model_cfg.get("model", "resnet18")
    num_classes = 7
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    y_true, y_pred, y_proba = [], [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="[test]"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).cpu().numpy().tolist()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred)
            y_proba.extend(probs.cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_proba, num_classes=num_classes)
    os.makedirs(args.out_dir, exist_ok=True)
    save_metrics_json(metrics, os.path.join(args.out_dir, "test_metrics.json"))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
