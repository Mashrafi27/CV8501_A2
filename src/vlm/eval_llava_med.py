
import argparse
import os
import csv
import json
from PIL import Image
from tqdm import tqdm

"""
Lightweight evaluation wrapper for LLaVA(-Med) style HF models.

This assumes your model supports:
  - A `processor` that can prepare (image, prompt) for generation
  - A `model.generate(...)` multimodal API

You MUST set `--model_id` to a local path or HF ID that works in your environment.
Examples:
  --model_id liuhaotian/llava-v1.5-7b-hf
  --model_id <local_path_to_llava_med_checkpoint>

NOTE: Some LLaVA-Med checkpoints use custom repos / launchers. If so, adapt this file.
"""

def load_model_and_processor(model_id: str, dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    t_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=t_dtype,
        device_map="auto",
        trust_remote_code=True
    )
    return model, processor

def ask_one(model, processor, image_path: str, question: str, max_new_tokens: int = 64):
    from transformers import GenerationConfig
    image = Image.open(image_path).convert("RGB")
    prompt = question
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return text

def extract_closed_answer(generated_text: str, options):
    # Heuristic: find the first option present in the text (case-insensitive)
    low = generated_text.lower()
    for opt in options:
        if opt.lower() in low:
            return opt
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_jsonl", required=True, type=str)
    ap.add_argument("--model_id", required=True, type=str)
    ap.add_argument("--out_csv", required=True, type=str)
    ap.add_argument("--dtype", default="bfloat16", type=str)
    args = ap.parse_args()

    model, processor = load_model_and_processor(args.model_id, dtype=args.dtype)

    # Read pairs
    pairs = []
    with open(args.pairs_jsonl, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
    options = ["akiec","bcc","bkl","df","nv","mel","vasc"]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "question", "gt_answer", "generated", "closed_answer"])
        for ex in tqdm(pairs, desc="eval llava-med"):
            gen = ask_one(model, processor, ex["image"], ex["question"])
            closed = extract_closed_answer(gen, options)
            writer.writerow([ex["image"], ex["question"], ex["answer"], gen, closed])

    print("Wrote predictions to", args.out_csv)

if __name__ == "__main__":
    main()
