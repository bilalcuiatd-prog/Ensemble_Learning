from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import max_tanimoto_to_train
from .config import ExperimentConfig, TargetConfig
from .features import build_feature_matrix
from .metrics import score_block
from .plots import (
    plot_ad_histogram,
    plot_confusion_matrix,
    plot_external_calibration,
    plot_external_curves,
    plot_score_distribution,
)
from .utils import save_dataframe, save_json


def evaluate_external_dataset(
    ext_df: pd.DataFrame,
    ext_stats: dict,
    label: str,
    cfg: TargetConfig,
    exp: ExperimentConfig,
    feature_selector,
    trained_full_base,
    final_meta_model,
    base_model_names: list[str],
    smiles_train,
    threshold_bacc: float,
    threshold_acc: float,
) -> dict:
    if len(ext_df) == 0:
        raise RuntimeError(f"External dataset '{label}' is empty after curation.")

    X_ext_raw, _ = build_feature_matrix(ext_df["smiles_std"].values, exp)
    X_ext = feature_selector.transform(X_ext_raw) if feature_selector is not None else X_ext_raw
    y_ext = ext_df["y_ext"].values.astype(int)

    P_ext = np.zeros((len(ext_df), len(trained_full_base)), dtype=np.float32)
    for j, (_, est) in enumerate(trained_full_base):
        P_ext[:, j] = est.predict_proba(X_ext)[:, 1]

    p_ext = final_meta_model.predict_proba(P_ext)[:, 1]

    bacc_metrics = score_block(y_ext, p_ext, threshold_bacc)
    acc_metrics = score_block(y_ext, p_ext, threshold_acc)
    summary_df = pd.DataFrame([
        {"threshold_type": "BACC-optimized", **bacc_metrics},
        {"threshold_type": "ACC-optimized", **acc_metrics},
    ])

    out = ext_df.copy()
    for j, name in enumerate(base_model_names):
        out[f"Prob_{name.upper()}"] = P_ext[:, j]
    out["Prob_STACK"] = p_ext
    out["Pred_STACK_thrBACC"] = (p_ext >= threshold_bacc).astype(int)
    out["Pred_STACK_thrACC"] = (p_ext >= threshold_acc).astype(int)

    similarities = max_tanimoto_to_train(smiles_train, out["smiles_std"].values, radius=exp.radius, n_bits=exp.n_bits)
    out["max_tanimoto_to_train"] = similarities
    out["in_domain"] = out["max_tanimoto_to_train"] >= exp.ad_similarity_threshold
    out["error_type"] = np.where(
        (out["y_ext"] == 0) & (out["Pred_STACK_thrBACC"] == 1),
        "FP",
        np.where((out["y_ext"] == 1) & (out["Pred_STACK_thrBACC"] == 0), "FN", "Correct"),
    )
    out["confidence_margin"] = np.abs(out["Prob_STACK"] - threshold_bacc)
    out["near_threshold"] = out["confidence_margin"] < 0.10

    error_summary = out.groupby("error_type").agg(
        n=("error_type", "size"),
        mean_prob=("Prob_STACK", "mean"),
        mean_similarity=("max_tanimoto_to_train", "mean"),
        near_threshold_fraction=("near_threshold", "mean"),
    ).reset_index()

    domain_rows = []
    for group_name, mask in [("In-domain", out["in_domain"].values), ("Out-of-domain", ~out["in_domain"].values)]:
        if mask.sum() > 0 and len(np.unique(y_ext[mask])) == 2:
            m = score_block(y_ext[mask], p_ext[mask], threshold_bacc)
            domain_rows.append({"subset": group_name, "n": int(mask.sum()), **m})
    domain_df = pd.DataFrame(domain_rows)

    label_slug = label.lower().replace(" ", "_").replace("-", "_")
    save_dataframe(summary_df, Path(cfg.output_dir) / f"external_{label_slug}_summary.csv")
    save_dataframe(out, Path(cfg.output_dir) / f"external_{label_slug}_predictions.csv")
    save_dataframe(error_summary, Path(cfg.output_dir) / f"external_{label_slug}_error_summary.csv")
    save_dataframe(domain_df, Path(cfg.output_dir) / f"external_{label_slug}_domain_performance.csv")
    save_json(ext_stats, Path(cfg.output_dir) / f"external_{label_slug}_construction_stats.json")

    plot_external_curves(y_ext, p_ext, cfg.target, label_slug, cfg.figure_dir)
    plot_external_calibration(y_ext, p_ext, cfg.target, label_slug, cfg.figure_dir)
    plot_score_distribution(y_ext, p_ext, threshold_bacc, cfg.target, label_slug, cfg.figure_dir)
    plot_confusion_matrix(y_ext, p_ext, threshold_bacc, cfg.target, f"{label_slug}_bacc", cfg.figure_dir)
    plot_confusion_matrix(y_ext, p_ext, threshold_acc, cfg.target, f"{label_slug}_acc", cfg.figure_dir)
    plot_ad_histogram(similarities, exp.ad_similarity_threshold, cfg.target, label_slug, cfg.figure_dir)

    return {
        "summary": summary_df,
        "predictions": out,
        "error_summary": error_summary,
        "domain_performance": domain_df,
    }
