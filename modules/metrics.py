from typing import Dict, Iterable, Optional

import numpy as np


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, int]:
    t = np.asarray(list(y_true), dtype=np.int64)
    p = np.asarray(list(y_pred), dtype=np.int64)
    tp = int(((t == 1) & (p == 1)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def binary_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    c = confusion_counts(y_true, y_pred)
    acc = safe_div(c["tp"] + c["tn"], c["tp"] + c["tn"] + c["fp"] + c["fn"])
    precision = safe_div(c["tp"], c["tp"] + c["fp"])
    recall = safe_div(c["tp"], c["tp"] + c["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def roc_auc_score_safe(y_true: Iterable[int], y_prob: Iterable[float]) -> Optional[float]:
    y_true_list = list(y_true)
    y_prob_list = list(y_prob)
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if len(set(y_true_list)) < 2:
            return None
        return float(roc_auc_score(y_true_list, y_prob_list))
    except Exception:
        return None

