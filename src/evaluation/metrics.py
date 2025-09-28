
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_metrics(y_true, y_pred, y_proba=None, num_classes=7):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    if y_proba is not None:
        # macro AUC (one-vs-rest)
        Y = label_binarize(y_true, classes=list(range(num_classes)))
        try:
            auc = roc_auc_score(Y, y_proba, average="macro", multi_class="ovr")
            out["auc_macro_ovr"] = float(auc)
        except Exception:
            out["auc_macro_ovr"] = None
    else:
        out["auc_macro_ovr"] = None
    return out

def save_metrics_json(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
