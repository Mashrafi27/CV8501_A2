
import argparse
import os
import json
import time
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd

from src.utils.seed import set_seed
from src.models.baseline_cnn import create_model, LabelSmoothingCrossEntropy
from src.data.transforms import build_transforms, IMAGENET_MEAN, IMAGENET_STD
from src.data.ham10000 import Ham10000Dataset
from src.evaluation.metrics import compute_metrics

@dataclass
class TrainConfig:
    train_csv: str
    val_csv: str
    image_size: int = 224
    model: str = "resnet18"
    pretrained: int = 1
    batch_size: int = 64
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    num_workers: int = 4
    out_dir: str = "./runs/baseline"
    seed: int = 42
    amp: int = 1

def accuracy_top1(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=str)
    ap.add_argument("--val_csv", required=True, type=str)
    ap.add_argument("--image_size", default=224, type=int)
    ap.add_argument("--model", default="resnet18", type=str)
    ap.add_argument("--pretrained", default=1, type=int)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--epochs", default=20, type=int)
    ap.add_argument("--lr", default=3e-4, type=float)
    ap.add_argument("--weight_decay", default=1e-4, type=float)
    ap.add_argument("--label_smoothing", default=0.0, type=float)
    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument("--out_dir", default="./runs/baseline", type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--amp", default=1, type=int, help="use torch.cuda.amp for mixed precision")
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load dataset-specific stats if present
    stats_path = os.path.join(os.path.dirname(os.path.dirname(cfg.train_csv)), "stats.json")
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    if os.path.exists(stats_path):
        try:
            s = json.load(open(stats_path))
            mean, std = s.get("mean", mean), s.get("std", std)
            print("Using dataset stats:", mean, std)
        except Exception:
            pass

    t_train = build_transforms(cfg.image_size, train=True, mean=mean, std=std)
    t_val = build_transforms(cfg.image_size, train=False, mean=mean, std=std)
    train_ds = Ham10000Dataset(cfg.train_csv, transform=t_train)
    val_ds = Ham10000Dataset(cfg.val_csv, transform=t_val)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    num_classes = 7
    model = create_model(cfg.model, num_classes=num_classes, pretrained=bool(cfg.pretrained))
    model.to(device)

    if cfg.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(eps=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")

    for epoch in range(cfg.epochs):
        model.train()
        run_loss, run_acc = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = accuracy_top1(logits.detach(), labels)
            run_loss += loss.item() * imgs.size(0)
            run_acc  += acc * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")
        optimizer.step()
        scheduler.step()

        train_loss = run_loss / len(train_ds)
        train_acc  = run_acc / len(train_ds)

        # ---- validation
        model.eval()
        y_true, y_pred, y_proba = [], [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="[val]"):
                imgs = imgs.to(device)
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1).cpu().numpy().tolist()
                y_true.extend(labels.numpy().tolist())
                y_pred.extend(pred)
                y_proba.extend(probs.cpu().numpy().tolist())

        from src.evaluation.metrics import compute_metrics
        metrics = compute_metrics(y_true, y_pred, y_proba, num_classes=num_classes)
        val_acc, val_f1 = metrics["accuracy"], metrics["f1_macro"]

        # log
        log_row = {
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_f1_macro": val_f1,
            "val_auc_macro_ovr": metrics["auc_macro_ovr"],
            "lr": scheduler.get_last_lr()[0],
        }
        print("Epoch summary:", log_row)

        # append to CSV log
        log_csv = os.path.join(cfg.out_dir, "val_log.csv")
        import csv
        exists = os.path.exists(log_csv)
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(log_row)

        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, best_path)
            print(f"Saved best model to {best_path} (val F1={best_val_f1:.4f})")

    print("Training complete. Best val F1:", best_val_f1)

if __name__ == "__main__":
    main()
