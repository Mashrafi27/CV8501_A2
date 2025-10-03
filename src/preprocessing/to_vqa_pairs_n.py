import argparse
import json
import pandas as pd
from pathlib import Path

HAM_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']
PROMPT = "What is the lesion type? Choose one of: {}.".format(", ".join(HAM_CLASSES))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True, type=str)
    ap.add_argument("--out_jsonl", required=True, type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv)

    img_col = "image_path" if "image_path" in df.columns else ("image" if "image" in df.columns else None)
    if img_col is None:
        raise KeyError("Expected 'image_path' or 'image' column")

    label_col = "label_name" if "label_name" in df.columns else ("label" if "label" in df.columns else None)
    if label_col is None:
        raise KeyError("Expected 'label_name' or 'label' column")

    with open(args.out_jsonl, "w") as f:
        for i, r in df.iterrows():
            image_filename = Path(str(r[img_col])).name  # <-- just filename
            ex = {
                "question_id": int(i),
                "image": image_filename,                 # filename only
                "text": PROMPT + "\n<image>",            # LLaVA expects <image>
                "answer": r[label_col]                   # keep for scoring
            }
            f.write(json.dumps(ex) + "\n")
    print("Wrote", args.out_jsonl)

if __name__ == "__main__":
    main()

