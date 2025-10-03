#!/usr/bin/env python3
import argparse, json, re, csv, sys
from collections import Counter, defaultdict
from pathlib import Path

HAM_CLASSES = ["akiec","bcc","bkl","df","nv","mel","vasc"]

# High-precision, case-insensitive regex patterns for each class.
# We avoid substring traps (e.g., "melanocytic" should map to 'nv', not 'mel').
PATTERNS = {
    "mel": [
        r"\bmelanoma(s)?\b",
        r"\bmalignant\s+melanoma\b",
    ],
    "bcc": [
        r"\b(basal\s+cell\s+carcinoma|basal[-\s]?cell\b.*carcinoma)\b",
        r"\bbcc\b",
    ],
    "akiec": [  # Actinic keratosis / Bowen (SCC in-situ)
        r"\bactinic\s+keratosis\b",
        r"\bbowen('?s)?\s+disease\b",
        r"\bsquamous\s+cell\s+carcinoma\s+(in\s+situ|insitu|is)\b",
        r"\bscc\s+(in\s+situ|insitu|is)\b",
        r"\bakiec\b",
    ],
    "nv": [
        r"\bmelanocytic\s+na?ev(us|i)\b",
        r"\bna?ev(us|i)\b",
        r"\bmelanocytic\b",  # keep after 'melanoma' checks
        r"\bnv\b",
    ],
    "bkl": [
        r"\bseborr?hoe?ic\s+keratosis\b",
        r"\bsolar\s+lentigo\b",
        r"\blichen\s+planus[-\s]?like\s+keratosis\b",
        r"\bbenign\s+keratosis\b",
        r"\bbkl\b",
    ],
    "df": [
        r"\bdermatofibroma(s)?\b",
        r"\bdf\b",
    ],
    "vasc": [
        r"\bvascular\s+lesion(s)?\b",
        r"\bangioma(s)?\b",
        r"\bangiokeratoma(s)?\b",
        r"\bpyogenic\s+granuloma\b",
        r"\bhemorrhage\b",
        r"\bcherry\s+angioma\b",
        r"\bvenous\s+lake\b",
        r"\bvasc\b",
    ],
}

# Compile regex
RX = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}

# Negation window: simple heuristic to skip "no melanoma", "not melanoma", "without melanoma"
NEG = re.compile(r"\b(no|not|without|absent|free\s+of)\b", re.I)

# Priority if multiple classes appear (rare but can happen). Malignancies first.
PRIORITY = ["mel","bcc","akiec","nv","bkl","df","vasc"]

def _is_negated(text: str, m: re.Match, window: int = 6) -> bool:
    # Check up to 'window' words before the match for a negation token.
    start = m.start()
    pre = text[:start]
    tokens = re.findall(r"\w+|\S", pre.lower())
    before = tokens[-window:]
    return any(NEG.fullmatch(tok) for tok in before)

def extract_label(answer_text: str) -> str | None:
    if not answer_text:
        return None
    t = answer_text.lower()
    found = []
    for cls in PRIORITY:
        for r in RX[cls]:
            for m in r.finditer(t):
                if not _is_negated(t, m):
                    found.append((cls, m.start(), m.group(0)))
                    break
            if any(x[0] == cls for x in found):
                break
    if not found:
        return None
    # Choose the one that appears earliest among highest priority already enforced
    found.sort(key=lambda x: (PRIORITY.index(x[0]), x[1]))
    return found[0][0]

def load_jsonl(path: str):
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                print(f"[WARN] bad json on line {i}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="test_vqa.jsonl with GT: keys include 'answer' and optionally 'question_id'")
    ap.add_argument("--answers", required=True, help="predictions jsonl from llava: keys include 'text' and 'question_id'")
    ap.add_argument("--out-csv", default=None, help="optional path to write per-sample details")
    ap.add_argument("--out-confusion", default=None, help="optional path to write confusion matrix csv")
    args = ap.parse_args()

    # Load GT
    q_records = list(load_jsonl(args.questions))
    # Build dict by question_id if present, else by index
    gt_by_qid = {}
    gt_seq = []
    for i, ex in q_records:
        qid = ex.get("question_id", i)
        gt = ex.get("answer")
        img = ex.get("image")
        gt_by_qid[qid] = (gt, img, ex)
        gt_seq.append((qid, gt, img, ex))

    # Load predictions
    a_records = list(load_jsonl(args.answers))
    pred_pairs = []
    matched = 0
    for i, ex in a_records:
        qid = ex.get("question_id", i)
        raw = ex.get("text") or ex.get("answer") or ex.get("response") or ""
        pred_pairs.append((qid, raw))

    # Evaluate
    details = []
    cov = 0
    correct = 0
    total = 0

    # Align by qid when available
    for qid, raw in pred_pairs:
        if qid not in gt_by_qid:
            continue
        gt, img, qex = gt_by_qid[qid]
        pred = extract_label(raw)
        total += 1
        if pred is not None:
            cov += 1
            if pred == gt:
                correct += 1
        details.append({
            "question_id": qid,
            "image": img,
            "gt": gt,
            "pred": pred,
            "raw_text": raw
        })

    # Metrics
    coverage = cov / total if total else 0.0
    accuracy = (correct / cov) if cov else 0.0
    print(f"Total matched: {total}")
    print(f"Coverage: {cov}/{total} = {coverage:.2%}")
    print(f"Closed accuracy: {correct}/{cov} = {accuracy:.2%}")

    # Confusion
    cm = defaultdict(lambda: Counter({c:0 for c in HAM_CLASSES}))
    for d in details:
        if d["pred"] is None or d["gt"] not in HAM_CLASSES:
            continue
        cm[d["gt"]][d["pred"]] += 1

    # Write outputs
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["question_id","image","gt","pred","raw_text"])
            w.writeheader()
            for d in details:
                w.writerow(d)
        print(f"Wrote per-sample: {args.out_csv}")

    if args.out_confusion:
        Path(args.out_confusion).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_confusion, "w", newline="") as f:
            cols = ["gt"] + HAM_CLASSES
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for gt in HAM_CLASSES:
                row = {"gt": gt}
                row.update({c: cm[gt][c] for c in HAM_CLASSES})
                w.writerow(row)
        print(f"Wrote confusion: {args.out_confusion}")

if __name__ == "__main__":
    main()

