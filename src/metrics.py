import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_classic_metrics(y_true, y_proba, threshold: float = 0.5):
    """
    Compute classic classification metrics for a given decision threshold.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1).
    y_proba : array-like
        Predicted probabilities for class 1.
    threshold : float, optional
        Decision threshold to convert probabilities into 0/1, by default 0.5.

    Returns
    -------
    dict
        Dictionary with AUC, precision, recall, f1.
    """
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
    """
    Compute business cost with different weights for FN and FP.

    FN (false negative)  = bad client predicted as good  -> very expensive.
    FP (false positive)  = good client predicted as bad  -> less expensive.

    Parameters
    ----------
    y_true : array-like
        True labels (0/1).
    y_proba : array-like
        Predicted probabilities for class 1.
    threshold : float
        Decision threshold.
    cost_fn : float
        Cost weight for each FN.
    cost_fp : float
        Cost weight for each FP.
    normalize : bool
        If True, divide total cost by number of samples.

    Returns
    -------
    total_cost : float
        Total (or average) business cost.
    conf : dict
        Confusion matrix counts: tn, fp, fn, tp.
    """
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
    """
    Compute cost for a list of thresholds to build the cost-vs-threshold curve.

    Returns
    -------
    thresholds : np.ndarray
    costs : np.ndarray
    """
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
