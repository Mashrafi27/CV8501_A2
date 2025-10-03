#!/usr/bin/env python3
import argparse, json, re, csv, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

HAM_CLASSES = ["akiec","bcc","bkl","df","nv","mel","vasc"]

# Regex patterns to extract labels from free-text responses
PATTERNS = {
    "mel": [
        r"\bmelanoma(s)?\b",
        r"\bmalignant\s+melanoma\b",
    ],
    "bcc": [
        r"\b(basal\s+cell\s+carcinoma|basal[-\s]?cell\b.*carcinoma)\b",
        r"\bbcc\b",
    ],
    "akiec": [
        r"\bactinic\s+keratosis\b",
        r"\bbowen('?s)?\s+disease\b",
        r"\bsquamous\s+cell\s+carcinoma\s+(in\s+situ|insitu|is)\b",
        r"\bscc\s+(in\s+situ|insitu|is)\b",
        r"\bakiec\b",
    ],
    "nv": [
        r"\bmelanocytic\s+na?ev(us|i)\b",
        r"\bna?ev(us|i)\b",
        r"\bmelanocytic\b",
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

RX = {k: [re.compile(p, re.I) for p in v] for k, v in PATTERNS.items()}
NEG = re.compile(r"\b(no|not|without|absent|free\s+of)\b", re.I)
PRIORITY = ["mel","bcc","akiec","nv","bkl","df","vasc"]

def _is_negated(text: str, m: re.Match, window: int = 6) -> bool:
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

def evaluate(gt_labels, pred_labels):
    cls2id = {c:i for i,c in enumerate(HAM_CLASSES)}
    y_true = [cls2id[g] for g,p in zip(gt_labels, pred_labels) if g in cls2id]
    y_pred = [cls2id[p] if p in cls2id else -1 for g,p in zip(gt_labels, pred_labels) if g in cls2id]

    # Drop unmatched predictions
    mask = [i for i, p in enumerate(y_pred) if p != -1]
    y_true = [y_true[i] for i in mask]
    y_pred = [y_pred[i] for i in mask]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # One-hot for AUC
    y_true_oh = np.eye(len(HAM_CLASSES))[y_true]
    y_pred_oh = np.eye(len(HAM_CLASSES))[y_pred]
    try:
        auc = roc_auc_score(y_true_oh, y_pred_oh, multi_class="ovr")
    except ValueError:
        auc = None

    print(f"\n=== Metrics ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 (macro): {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    if auc:
        print(f"AUC (OvR): {auc:.4f}")
    else:
        print("AUC could not be computed (not enough class coverage)")
    return acc, f1_macro, f1_weighted, auc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Ground truth jsonl (with 'answer')")
    ap.add_argument("--answers", required=True, help="Predictions jsonl (with 'text')")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-confusion", default=None)
    args = ap.parse_args()

    q_records = list(load_jsonl(args.questions))
    gt_by_qid = {}
    for i, ex in q_records:
        qid = ex.get("question_id", i)
        gt_by_qid[qid] = ex.get("answer")

    a_records = list(load_jsonl(args.answers))
    details = []
    for i, ex in a_records:
        qid = ex.get("question_id", i)
        raw = ex.get("text") or ex.get("answer") or ex.get("response") or ""
        gt = gt_by_qid.get(qid)
        pred = extract_label(raw)
        details.append({"qid": qid, "gt": gt, "pred": pred, "raw": raw})

    gt_labels = [d["gt"] for d in details if d["gt"] in HAM_CLASSES]
    pred_labels = [d["pred"] for d in details if d["gt"] in HAM_CLASSES]

    evaluate(gt_labels, pred_labels)

if __name__ == "__main__":
    main()

