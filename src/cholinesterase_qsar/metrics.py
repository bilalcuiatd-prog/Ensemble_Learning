from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


def safe_roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_pr_auc(y, p) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def ece_mce(y_true, p_pred, n_bins: int = 10) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pred >= lo) & ((p_pred < hi) if i < n_bins - 1 else (p_pred <= hi))
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(p_pred[mask].mean())
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(y_true))
        mce = max(mce, gap)
    return float(ece), float(mce)


def choose_best_threshold(y_true, p_pred, metric: str = "balanced_accuracy") -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_val = 0.5, -1.0
    for threshold in thresholds:
        yhat = (p_pred >= threshold).astype(int)
        if metric == "accuracy":
            value = accuracy_score(y_true, yhat)
        else:
            value = balanced_accuracy_score(y_true, yhat)
        if value > best_val:
            best_val, best_t = value, threshold
    return float(best_t), float(best_val)


def score_block(y_true, p_pred, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)
    yhat = (p_pred >= threshold).astype(int)
    ece, mce = ece_mce(y_true, p_pred, n_bins=10)
    cm = confusion_matrix(y_true, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "ACC": float(accuracy_score(y_true, yhat)),
        "BACC": float(balanced_accuracy_score(y_true, yhat)),
        "MCC": float(matthews_corrcoef(y_true, yhat)),
        "ROC_AUC": safe_roc_auc(y_true, p_pred),
        "PR_AUC": safe_pr_auc(y_true, p_pred),
        "Brier": float(brier_score_loss(y_true, p_pred)),
        "LogLoss": float(log_loss(y_true, np.clip(p_pred, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ECE": ece,
        "MCE": mce,
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
