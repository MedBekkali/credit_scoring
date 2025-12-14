import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classic_metrics(y_true, y_proba, threshold: float = 0.5):
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def business_cost(
    y_true,
    y_proba,
    threshold: float = 0.5,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    normalize: bool = True,
):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    total_cost = cost_fn * fn + cost_fp * fp

    if normalize:
        total_cost = total_cost / len(y_true)

    return total_cost, {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def cost_curve(
    y_true,
    y_proba,
    thresholds=None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    normalize: bool = True,
):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    costs = []
    for thr in thresholds:
        c, _ = business_cost(
            y_true,
            y_proba,
            threshold=thr,
            cost_fn=cost_fn,
            cost_fp=cost_fp,
            normalize=normalize,
        )
        costs.append(c)

    return np.array(thresholds), np.array(costs)