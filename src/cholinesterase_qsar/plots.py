from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve


def configure_plots() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (7, 4),
        "axes.spines.right": False,
        "axes.spines.top": False,
    })


def savefig(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_summary(summary_df: pd.DataFrame, target: str, figure_dir: Path) -> None:
    plt.figure(figsize=(8, 4))
    x = np.arange(len(summary_df))
    plt.errorbar(x, summary_df["mean"], yerr=summary_df["std"], fmt="o", capsize=5)
    plt.xticks(x, summary_df["metric"], rotation=25, ha="right")
    plt.ylabel("Value")
    plt.title(f"{target}: Nested Scaffold CV Mean ± SD")
    savefig(Path(figure_dir) / "nested_cv_mean_sd.png")


def plot_cv_curves(roc_curves, pr_curves, target: str, figure_dir: Path) -> None:
    plt.figure(figsize=(6, 5))
    for i, (fpr, tpr) in enumerate(roc_curves, start=1):
        plt.plot(fpr, tpr, label=f"Fold {i}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{target}: ROC Curves Across Outer Scaffold Folds")
    plt.legend(loc="lower right", fontsize=8)
    savefig(Path(figure_dir) / "nested_cv_roc_folds.png")

    plt.figure(figsize=(6, 5))
    for i, (rec, prec) in enumerate(pr_curves, start=1):
        plt.plot(rec, prec, label=f"Fold {i}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{target}: PR Curves Across Outer Scaffold Folds")
    plt.legend(loc="lower left", fontsize=8)
    savefig(Path(figure_dir) / "nested_cv_pr_folds.png")


def plot_calibration_comparison(y, p_uncal, p_cal, target: str, figure_dir: Path) -> None:
    frac_pos_u, mean_pred_u = calibration_curve(y, p_uncal, n_bins=10, strategy="uniform")
    frac_pos_c, mean_pred_c = calibration_curve(y, p_cal, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred_u, frac_pos_u, marker="o", label="Uncalibrated")
    plt.plot(mean_pred_c, frac_pos_c, marker="o", label="Sigmoid-calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"{target}: Reliability Plot")
    plt.legend()
    savefig(Path(figure_dir) / "calibration_uncalibrated_vs_calibrated.png")


def plot_external_curves(y, p, target: str, label: str, figure_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"Final stack (ROC-AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{target}: {label} ROC Curve")
    plt.legend(loc="lower right")
    savefig(Path(figure_dir) / f"external_{label.lower()}_roc.png")

    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, linewidth=2, label=f"Final stack (PR-AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{target}: {label} Precision-Recall Curve")
    plt.legend(loc="lower left")
    savefig(Path(figure_dir) / f"external_{label.lower()}_pr.png")


def plot_external_calibration(y, p, target: str, label: str, figure_dir: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title(f"{target}: {label} Reliability Plot")
    plt.legend()
    savefig(Path(figure_dir) / f"external_{label.lower()}_calibration.png")


def plot_score_distribution(y, p, threshold: float, target: str, label: str, figure_dir: Path) -> None:
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(p[np.asarray(y) == 1], bins=20, alpha=0.7, label="Active")
    plt.hist(p[np.asarray(y) == 0], bins=20, alpha=0.7, label="Inactive")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold = {threshold:.3f}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(f"{target}: {label} Score Distribution")
    plt.legend()
    savefig(Path(figure_dir) / f"external_{label.lower()}_score_distribution.png")


def plot_confusion_matrix(y, p, threshold: float, target: str, label: str, figure_dir: Path) -> None:
    y_pred = (np.asarray(p) >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["Inactive", "Active"],
        yticklabels=["Inactive", "Active"],
        xlabel="Predicted",
        ylabel="Actual",
        title=f"{target}: {label} Confusion Matrix",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", fontsize=11)
    savefig(Path(figure_dir) / f"external_{label.lower()}_confusion_matrix.png")


def plot_ad_histogram(similarities, threshold: float, target: str, label: str, figure_dir: Path) -> None:
    plt.figure(figsize=(6, 3.8))
    plt.hist(similarities, bins=20)
    plt.axvline(threshold, linestyle="--", linewidth=2)
    plt.xlabel("Max Tanimoto to internal training set")
    plt.ylabel("Count")
    plt.title(f"{target}: {label} Applicability Domain")
    savefig(Path(figure_dir) / f"external_{label.lower()}_ad_similarity.png")
