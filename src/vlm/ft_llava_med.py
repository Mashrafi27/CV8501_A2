
import argparse
import os
import json
from dataclasses import dataclass
from typing import Optional

"""
LoRA fine-tuning skeleton for LLaVA(-Med) style HF models.
This is intentionally minimal and may require adaptation to the specific checkpoint you use.

If your LLaVA-Med model requires a custom training harness, consider using that project
directly and just point it to the VQA pairs produced by `to_vqa_pairs.py`.
"""

@dataclass
class FTConfig:
    train_pairs: str
    val_pairs: str
    base_model_id: str
    output_dir: str = "./runs/llava_med_lora"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum_steps: int = 16
    epochs: int = 1
    dtype: str = "bfloat16"
    max_new_tokens: int = 64

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pairs", required=True, type=str)
    ap.add_argument("--val_pairs", required=True, type=str)
    ap.add_argument("--base_model_id", required=True, type=str)
    ap.add_argument("--output_dir", default="./runs/llava_med_lora", type=str)
    ap.add_argument("--lora_r", default=8, type=int)
    ap.add_argument("--lora_alpha", default=16, type=int)
    ap.add_argument("--lora_dropout", default=0.05, type=float)
    ap.add_argument("--lr", default=2e-5, type=float)
    ap.add_argument("--batch_size", default=2, type=int)
    ap.add_argument("--grad_accum_steps", default=16, type=int)
    ap.add_argument("--epochs", default=1, type=int)
    ap.add_argument("--dtype", default="bfloat16", type=str)
    ap.add_argument("--max_new_tokens", default=64, type=int)
    args = ap.parse_args()
    cfg = FTConfig(**vars(args))

    os.makedirs(cfg.output_dir, exist_ok=True)
    # We provide a placeholder to encourage using the official LLaVA(-Med) training harness.
    # If you want to finish this here, you can integrate PEFT.LoRA + transformers Trainer
    # with a collator that loads (image, question) -> text target.
    raise NotImplementedError(
        "LoRA fine-tuning is a skeleton. Use the official LLaVA(-Med) training harness, "
        "or extend this file to your checkpoint."
    )

if __name__ == "__main__":
    main()
