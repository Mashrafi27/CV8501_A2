#!/bin/bash

#SBATCH -p nvidia
#SBATCH --gres=gpu:1

python -m src.training.train_baseline --train_csv ./data/splits/train.csv --val_csv   ./data/splits/val.csv --image_size 224 --model resnet18 --pretrained 1 --batch_size 64 --epochs 20 --lr 3e-4 --weight_decay 1e-4 --out_dir ./runs/baseline_resnet18

