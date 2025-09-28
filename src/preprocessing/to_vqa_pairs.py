
import argparse
import json
import pandas as pd

HAM_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'mel', 'vasc']

PROMPT = "What is the lesion type? Choose one of: {}.".format(", ".join(HAM_CLASSES))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_csv", required=True, type=str)
    ap.add_argument("--out_jsonl", required=True, type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.split_csv)
    with open(args.out_jsonl, "w") as f:
        for _, r in df.iterrows():
            ex = {
                "image": r["image_path"],
                "question": PROMPT,
                "answer": r["label_name"]
            }
            f.write(json.dumps(ex) + "\n")
    print("Wrote", args.out_jsonl)

if __name__ == "__main__":
    main()
