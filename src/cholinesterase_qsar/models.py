from __future__ import annotations

import time

import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from .config import ExperimentConfig
from .metrics import choose_best_threshold


def get_base_models(y, exp: ExperimentConfig):
    pos = int(np.sum(y))
    neg = int(len(y) - pos)
    scale_pos_weight = neg / max(pos, 1)
    lgbm_params = dict(
        n_estimators=exp.lgbm_estimators,
        learning_rate=0.01,
        num_leaves=127,
        max_depth=10,
        min_child_samples=60,
        min_split_gain=0.05,
        reg_alpha=1.5,
        reg_lambda=10.0,
        colsample_bytree=0.75,
        subsample=0.8,
        subsample_freq=1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=exp.random_state,
    )
    return [
        ("lgbm", LGBMClassifier(**lgbm_params)),
        ("rf", RandomForestClassifier(
            n_estimators=1200,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=exp.random_state,
        )),
        ("etc", ExtraTreesClassifier(
            n_estimators=1600,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=exp.random_state,
        )),
        ("svd_lr", Pipeline([
            ("svd", TruncatedSVD(n_components=384, random_state=exp.random_state)),
            ("lr", LogisticRegression(C=0.15, class_weight="balanced", max_iter=8000, random_state=exp.random_state)),
        ])),
    ]


def get_meta_candidates(exp: ExperimentConfig) -> dict:
    return {
        "lr": LogisticRegression(C=0.05, class_weight="balanced", max_iter=12000, random_state=exp.random_state),
        "rf": RandomForestClassifier(n_estimators=500, max_depth=4, class_weight="balanced", random_state=exp.random_state, n_jobs=-1),
        "lgbm": LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, max_depth=4, random_state=exp.random_state, n_jobs=-1),
    }


def fit_with_optional_early_stopping(est, X_tr, y_tr, X_va=None, y_va=None, exp: ExperimentConfig | None = None):
    stopping_rounds = 300 if exp is None else exp.lgbm_early_stopping_rounds
    if isinstance(est, LGBMClassifier) and X_va is not None and len(np.unique(y_va)) == 2:
        est.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=stopping_rounds, verbose=False), log_evaluation(period=0)],
        )
    else:
        est.fit(X_tr, y_tr)
    return est


def fit_full_with_internal_val(est, X, y, seed: int, exp: ExperimentConfig):
    if isinstance(est, LGBMClassifier) and len(np.unique(y)) == 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=seed)
        tr_i, va_i = next(sss.split(X, y))
        est.fit(
            X[tr_i],
            y[tr_i],
            eval_set=[(X[va_i], y[va_i])],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=exp.lgbm_early_stopping_rounds, verbose=False), log_evaluation(period=0)],
        )
    else:
        est.fit(X, y)
    return est


def build_base_oof_and_test(X_tr, y_tr, X_te, base_models, n_splits: int, seed: int, exp: ExperimentConfig):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_train, n_test = X_tr.shape[0], X_te.shape[0]
    oof = np.zeros((n_train, len(base_models)), dtype=np.float32)
    te = np.zeros((n_test, len(base_models)), dtype=np.float32)
    fit_times = {}
    fitted_full = []

    for j, (name, est) in enumerate(base_models):
        oof_j = np.zeros(n_train, dtype=np.float32)
        t0 = time.perf_counter()
        for tr_i, va_i in skf.split(X_tr, y_tr):
            est_fold = clone(est)
            est_fold = fit_with_optional_early_stopping(est_fold, X_tr[tr_i], y_tr[tr_i], X_tr[va_i], y_tr[va_i], exp=exp)
            oof_j[va_i] = est_fold.predict_proba(X_tr[va_i])[:, 1]
        fit_times[f"{name}_oof_fit_sec"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        est_full = clone(est)
        est_full = fit_full_with_internal_val(est_full, X_tr, y_tr, seed=seed + 100 + j, exp=exp)
        te[:, j] = est_full.predict_proba(X_te)[:, 1]
        fit_times[f"{name}_full_fit_plus_pred_sec"] = time.perf_counter() - t1
        fitted_full.append((name, est_full))
        oof[:, j] = oof_j

    return oof, te, fit_times, fitted_full


def threshold_from_inner_cv(Z, y, meta_estimator, calibrated: bool, calibrator_cv: int, inner_folds: int, metric: str, seed: int):
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=np.float32)
    for tr_i, va_i in skf.split(Z, y):
        est = clone(meta_estimator)
        if calibrated:
            est = CalibratedClassifierCV(est, method="sigmoid", cv=calibrator_cv)
        est.fit(Z[tr_i], y[tr_i])
        oof_prob[va_i] = est.predict_proba(Z[va_i])[:, 1]
    threshold, value = choose_best_threshold(y, oof_prob, metric=metric)
    return threshold, value, oof_prob
