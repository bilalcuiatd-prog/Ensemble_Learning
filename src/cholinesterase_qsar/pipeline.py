from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.pipeline import Pipeline

from .chemistry import max_tanimoto_to_train
from .config import ExperimentConfig, TargetConfig
from .data import (
    load_bindingdb_external_dataset,
    load_internal_dataset,
    load_labeled_external_dataset,
    load_pubchem_external_dataset,
)
from .external import evaluate_external_dataset
from .features import TrainOnlyFeatureSelector, build_feature_matrix
from .metrics import score_block
from .models import (
    build_base_oof_and_test,
    fit_full_with_internal_val,
    fit_with_optional_early_stopping,
    get_base_models,
    get_meta_candidates,
    threshold_from_inner_cv,
)
from .plots import configure_plots, plot_calibration_comparison, plot_cv_curves, plot_cv_summary
from .splitting import assert_no_inchikey_overlap, assert_no_scaffold_overlap, scaffold_stratified_kfold_indices
from .utils import ensure_dirs, save_dataframe, save_json, set_seed


def run_preprocessing_ablation(df_path: Path, exp: ExperimentConfig, output_dir: Path) -> pd.DataFrame:
    rows = []
    for mode in ["basic", "largest_fragment", "uncharged", "full"]:
        try:
            df_ab, _ = load_internal_dataset(df_path, standardize_mode=mode)
            y_ab = df_ab["is_active"].values.astype(int)
            scaf_ab = df_ab["scaffold"].values
            inch_ab = df_ab["inchikey"].values
            X_ab, _ = build_feature_matrix(df_ab["smiles_std"].values, exp)
            n_components = min(384, max(2, X_ab.shape[1] - 1), max(2, X_ab.shape[0] - 2))
            model = Pipeline([
                ("svd", TruncatedSVD(n_components=n_components, random_state=exp.random_state)),
                ("lr", LogisticRegression(C=0.15, class_weight="balanced", max_iter=6000, random_state=exp.random_state)),
            ])
            preds = np.zeros(len(y_ab), dtype=float)
            for idx_tr, idx_te in scaffold_stratified_kfold_indices(scaf_ab, y_ab, n_splits=exp.outer_folds):
                assert_no_inchikey_overlap(idx_tr, idx_te, inch_ab)
                assert_no_scaffold_overlap(idx_tr, idx_te, scaf_ab)
                model_fold = clone(model)
                model_fold.fit(X_ab[idx_tr], y_ab[idx_tr])
                preds[idx_te] = model_fold.predict_proba(X_ab[idx_te])[:, 1]
            threshold = 0.5
            scored = score_block(y_ab, preds, threshold)
            rows.append({
                "mode": mode,
                "n_after_dedup": len(df_ab),
                "n_unique_scaffolds": int(df_ab["scaffold"].nunique()),
                "roc_auc": scored["ROC_AUC"],
                "pr_auc": scored["PR_AUC"],
                "brier": scored["Brier"],
                "bacc": scored["BACC"],
                "mcc": scored["MCC"],
            })
        except Exception as exc:
            rows.append({"mode": mode, "error": str(exc)})
    df = pd.DataFrame(rows)
    save_dataframe(df, Path(output_dir) / "preprocessing_ablation.csv")
    return df


def run_pipeline(cfg: TargetConfig, exp: ExperimentConfig) -> dict:
    configure_plots()
    set_seed(exp.random_state)
    ensure_dirs(cfg.output_dir, cfg.figure_dir, cfg.model_dir)
    t_start = time.perf_counter()

    print(f"\n=== Target: {cfg.target} ===")
    df, internal_stats = load_internal_dataset(cfg.train_csv, standardize_mode="full")
    save_dataframe(pd.DataFrame([internal_stats]), Path(cfg.output_dir) / "internal_curation_stats.csv")
    save_dataframe(df[["canonical_smiles", "smiles_std", "inchikey", "scaffold", "is_active"]], Path(cfg.output_dir) / "internal_curated_compounds.csv")

    y_all = df["is_active"].values.astype(int)
    smiles_all = df["smiles_std"].values
    inchikeys = df["inchikey"].values
    scaffolds = df["scaffold"].values

    print("Building molecular feature matrix...")
    t_feat = time.perf_counter()
    X_all, feature_names = build_feature_matrix(smiles_all, exp)
    feature_build_sec = time.perf_counter() - t_feat
    pd.Series(feature_names, name="feature_name").to_csv(Path(cfg.output_dir) / "feature_names.csv", index=False)

    if exp.run_preprocess_ablation:
        print("Running preprocessing ablation...")
        run_preprocessing_ablation(cfg.train_csv, exp, cfg.output_dir)

    print("Running nested scaffold CV...")
    outer_rows, runtime_rows, roc_curves, pr_curves = [], [], [], []
    meta_candidates = get_meta_candidates(exp)

    for fold_id, (idx_tr, idx_te) in enumerate(scaffold_stratified_kfold_indices(scaffolds, y_all, n_splits=exp.outer_folds), start=1):
        assert_no_inchikey_overlap(idx_tr, idx_te, inchikeys)
        assert_no_scaffold_overlap(idx_tr, idx_te, scaffolds)

        X_tr_raw, X_te_raw = X_all[idx_tr], X_all[idx_te]
        y_tr, y_te = y_all[idx_tr], y_all[idx_te]
        smi_tr, smi_te = smiles_all[idx_tr], smiles_all[idx_te]
        scaf_tr, scaf_te = scaffolds[idx_tr], scaffolds[idx_te]
        fold_t0 = time.perf_counter()

        if exp.use_feature_selection:
            selector = TrainOnlyFeatureSelector(exp.variance_threshold, exp.top_k_features, exp.random_state + fold_id)
            selector.fit(X_tr_raw, y_tr)
            X_tr = selector.transform(X_tr_raw)
            X_te = selector.transform(X_te_raw)
            selected_features_n = selector.n_selected_
            after_variance_n = selector.n_after_variance_
        else:
            X_tr, X_te = X_tr_raw, X_te_raw
            selected_features_n = X_tr.shape[1]
            after_variance_n = X_tr.shape[1]

        base_models = get_base_models(y_tr, exp)
        oof_base, te_base, fit_times, _ = build_base_oof_and_test(
            X_tr, y_tr, X_te, base_models, n_splits=exp.inner_folds_oof, seed=exp.random_state + fold_id, exp=exp
        )
        Z_tr, Z_te = oof_base.astype(np.float32), te_base.astype(np.float32)

        best_key = None
        best_bacc, best_brier = -np.inf, np.inf
        for meta_name, meta_est in meta_candidates.items():
            for calibrated in [False, True]:
                thr_bacc, _, _ = threshold_from_inner_cv(
                    Z_tr, y_tr, meta_est, calibrated, exp.calibration_cv, exp.inner_folds_threshold,
                    "balanced_accuracy", exp.random_state + 1000 + fold_id
                )
                if calibrated:
                    final_meta = CalibratedClassifierCV(clone(meta_est), method="sigmoid", cv=exp.calibration_cv)
                else:
                    final_meta = clone(meta_est)
                final_meta.fit(Z_tr, y_tr)
                p_te = final_meta.predict_proba(Z_te)[:, 1]
                metrics = score_block(y_te, p_te, thr_bacc)
                if metrics["BACC"] > best_bacc or (abs(metrics["BACC"] - best_bacc) < 1e-12 and metrics["Brier"] < best_brier):
                    best_bacc, best_brier = metrics["BACC"], metrics["Brier"]
                    best_key = (meta_name, calibrated, thr_bacc, p_te, metrics)

        meta_name, calibrated, thr_bacc, p_te, fold_metrics = best_key
        max_sims_fold = max_tanimoto_to_train(smi_tr, smi_te, radius=exp.radius, n_bits=exp.n_bits)

        outer_rows.append({
            "fold": fold_id,
            "meta_selected": meta_name,
            "meta_calibrated": calibrated,
            "n_train": len(idx_tr),
            "n_test": len(idx_te),
            "train_pos_rate": float(y_tr.mean()),
            "test_pos_rate": float(y_te.mean()),
            "train_unique_scaffolds": len(set(scaf_tr)),
            "test_unique_scaffolds": len(set(scaf_te)),
            "median_max_train_test_tanimoto": float(np.median(max_sims_fold)),
            "mean_max_train_test_tanimoto": float(np.mean(max_sims_fold)),
            "fraction_test_out_of_domain_simlt035": float(np.mean(max_sims_fold < exp.ad_similarity_threshold)),
            "n_features_input": X_tr_raw.shape[1],
            "n_features_after_variance": after_variance_n,
            "n_features_selected": selected_features_n,
            "fold_runtime_sec": time.perf_counter() - fold_t0,
            **{k.lower().replace(" ", "_"): v for k, v in fold_metrics.items() if k != "TN_FP_FN_TP"},
        })
        runtime_rows.append({"fold": fold_id, **fit_times, "total_fold_sec": time.perf_counter() - fold_t0})
        fpr, tpr, _ = roc_curve(y_te, p_te)
        prec, rec, _ = precision_recall_curve(y_te, p_te)
        roc_curves.append((fpr, tpr))
        pr_curves.append((rec, prec))
        print(f"Fold {fold_id}: meta={meta_name}, calibrated={calibrated}, ROC-AUC={fold_metrics['ROC_AUC']:.4f}, BACC={fold_metrics['BACC']:.4f}")

    cv_df = pd.DataFrame(outer_rows)
    runtime_df = pd.DataFrame(runtime_rows)
    metric_cols = ["roc_auc", "pr_auc", "brier", "ece", "acc", "bacc", "mcc", "sensitivity", "specificity", "fold_runtime_sec"]
    summary_df = pd.DataFrame([
        {"metric": col, "mean": float(cv_df[col].mean()), "std": float(cv_df[col].std(ddof=1)) if len(cv_df) > 1 else 0.0}
        for col in metric_cols if col in cv_df.columns
    ])
    save_dataframe(cv_df, Path(cfg.output_dir) / "nested_cv_fold_results.csv")
    save_dataframe(summary_df, Path(cfg.output_dir) / "nested_cv_summary.csv")
    save_dataframe(runtime_df, Path(cfg.output_dir) / "runtime_breakdown.csv")
    plot_cv_summary(summary_df, cfg.target, cfg.figure_dir)
    plot_cv_curves(roc_curves, pr_curves, cfg.target, cfg.figure_dir)

    print("Training full-data stacked model...")
    if exp.use_feature_selection:
        global_selector = TrainOnlyFeatureSelector(exp.variance_threshold, exp.top_k_features, exp.random_state)
        global_selector.fit(X_all, y_all)
        X_all_sel = global_selector.transform(X_all)
    else:
        global_selector = None
        X_all_sel = X_all

    base_model_names = [name for name, _ in get_base_models(y_all, exp)]
    Z_oof_full = np.zeros((len(y_all), len(base_model_names)), dtype=np.float32)
    for j, (name, est) in enumerate(get_base_models(y_all, exp)):
        preds_j = np.zeros(len(y_all), dtype=np.float32)
        for idx_tr, idx_va in scaffold_stratified_kfold_indices(scaffolds, y_all, n_splits=exp.outer_folds):
            assert_no_inchikey_overlap(idx_tr, idx_va, inchikeys)
            assert_no_scaffold_overlap(idx_tr, idx_va, scaffolds)
            est_fold = clone(est)
            est_fold = fit_with_optional_early_stopping(est_fold, X_all_sel[idx_tr], y_all[idx_tr], X_all_sel[idx_va], y_all[idx_va], exp=exp)
            preds_j[idx_va] = est_fold.predict_proba(X_all_sel[idx_va])[:, 1]
        Z_oof_full[:, j] = preds_j

    meta_rows = []
    best_meta_name, best_meta_calibrated = None, None
    best_bacc, best_brier = -np.inf, np.inf
    for meta_name, meta_est in meta_candidates.items():
        for calibrated in [False, True]:
            thr_bacc_full, _, oof_prob = threshold_from_inner_cv(
                Z_oof_full, y_all, meta_est, calibrated, exp.calibration_cv, exp.inner_folds_threshold,
                "balanced_accuracy", exp.random_state + 5000
            )
            thr_acc_full, _, _ = threshold_from_inner_cv(
                Z_oof_full, y_all, meta_est, calibrated, exp.calibration_cv, exp.inner_folds_threshold,
                "accuracy", exp.random_state + 5001
            )
            m = score_block(y_all, oof_prob, thr_bacc_full)
            meta_rows.append({"meta": meta_name, "calibrated": calibrated, "thr_bacc": thr_bacc_full, "thr_acc": thr_acc_full, **m})
            if m["BACC"] > best_bacc or (abs(m["BACC"] - best_bacc) < 1e-12 and m["Brier"] < best_brier):
                best_bacc, best_brier = m["BACC"], m["Brier"]
                best_meta_name, best_meta_calibrated = meta_name, calibrated

    meta_df = pd.DataFrame(meta_rows).sort_values(["BACC", "Brier"], ascending=[False, True]).reset_index(drop=True)
    save_dataframe(meta_df, Path(cfg.output_dir) / "meta_learner_comparison.csv")

    final_meta_base = clone(meta_candidates[best_meta_name])
    final_meta_model = CalibratedClassifierCV(final_meta_base, method="sigmoid", cv=exp.calibration_cv) if best_meta_calibrated else final_meta_base
    final_meta_model.fit(Z_oof_full, y_all)

    thr_bacc_final, _, oof_prob_final = threshold_from_inner_cv(
        Z_oof_full, y_all, meta_candidates[best_meta_name], best_meta_calibrated, exp.calibration_cv,
        exp.inner_folds_threshold, "balanced_accuracy", exp.random_state + 6000
    )
    thr_acc_final, _, _ = threshold_from_inner_cv(
        Z_oof_full, y_all, meta_candidates[best_meta_name], best_meta_calibrated, exp.calibration_cv,
        exp.inner_folds_threshold, "accuracy", exp.random_state + 6001
    )

    if exp.run_calibration_analysis:
        _, _, uncal_prob = threshold_from_inner_cv(
            Z_oof_full, y_all, meta_candidates[best_meta_name], False, exp.calibration_cv,
            exp.inner_folds_threshold, "balanced_accuracy", exp.random_state + 6100
        )
        _, _, cal_prob = threshold_from_inner_cv(
            Z_oof_full, y_all, meta_candidates[best_meta_name], True, exp.calibration_cv,
            exp.inner_folds_threshold, "balanced_accuracy", exp.random_state + 6101
        )
        plot_calibration_comparison(y_all, uncal_prob, cal_prob, cfg.target, cfg.figure_dir)

    trained_full_base = []
    for j, (name, est) in enumerate(get_base_models(y_all, exp)):
        est_full = clone(est)
        est_full = fit_full_with_internal_val(est_full, X_all_sel, y_all, seed=exp.random_state + 7000 + j, exp=exp)
        trained_full_base.append((name, est_full))

    artifact = {
        "target_config": cfg.to_dict(),
        "experiment_config": exp.to_dict(),
        "feature_names": feature_names,
        "feature_selector": global_selector,
        "base_model_names": base_model_names,
        "trained_full_base": trained_full_base,
        "final_meta_model": final_meta_model,
        "best_meta_name": best_meta_name,
        "best_meta_calibrated": best_meta_calibrated,
        "thr_bacc_final": thr_bacc_final,
        "thr_acc_final": thr_acc_final,
        "internal_smiles": smiles_all,
        "internal_inchikeys": inchikeys,
        "internal_scaffolds": scaffolds,
        "internal_stats": internal_stats,
    }
    joblib.dump(artifact, Path(cfg.model_dir) / "final_artifact.joblib")
    save_json({
        "target": cfg.target,
        "internal_stats": internal_stats,
        "meta_selection": {"meta": best_meta_name, "calibrated": best_meta_calibrated},
        "thresholds": {"thr_bacc": thr_bacc_final, "thr_acc": thr_acc_final},
        "feature_build_sec": feature_build_sec,
        "total_runtime_sec": time.perf_counter() - t_start,
    }, Path(cfg.output_dir) / "run_summary.json")

    external_outputs = {}
    if cfg.bindingdb_tsv is not None and Path(cfg.bindingdb_tsv).exists():
        print("Evaluating BindingDB-style external set...")
        ext_df, ext_stats = load_bindingdb_external_dataset(cfg, inchikeys, scaffolds, standardize_mode="full")
        external_outputs["bindingdb"] = evaluate_external_dataset(
            ext_df, ext_stats, "BindingDB", cfg, exp, global_selector, trained_full_base, final_meta_model,
            base_model_names, smiles_all, thr_bacc_final, thr_acc_final
        )
    if cfg.pubchem_csv is not None and Path(cfg.pubchem_csv).exists():
        print("Evaluating PubChem-style external set...")
        ext_df, ext_stats = load_pubchem_external_dataset(cfg.pubchem_csv, cfg, inchikeys, scaffolds, standardize_mode="full")
        external_outputs["pubchem"] = evaluate_external_dataset(
            ext_df, ext_stats, "PubChem", cfg, exp, global_selector, trained_full_base, final_meta_model,
            base_model_names, smiles_all, thr_bacc_final, thr_acc_final
        )
    if cfg.balanced_external_csv is not None and Path(cfg.balanced_external_csv).exists():
        print("Evaluating balanced labeled external set...")
        ext_df, ext_stats = load_labeled_external_dataset(cfg.balanced_external_csv, inchikeys, scaffolds, standardize_mode="full")
        external_outputs["balanced"] = evaluate_external_dataset(
            ext_df, ext_stats, "Balanced", cfg, exp, global_selector, trained_full_base, final_meta_model,
            base_model_names, smiles_all, thr_bacc_final, thr_acc_final
        )

    print(f"\nPipeline completed in {time.perf_counter() - t_start:.2f} seconds.")
    return {
        "cv_results": cv_df,
        "cv_summary": summary_df,
        "meta_comparison": meta_df,
        "artifact_path": Path(cfg.model_dir) / "final_artifact.joblib",
        "external_outputs": external_outputs,
    }
