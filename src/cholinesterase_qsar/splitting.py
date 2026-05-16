from __future__ import annotations

from collections import defaultdict

import numpy as np


def assert_no_inchikey_overlap(train_idx, test_idx, inchikeys) -> None:
    tr = set(np.asarray(inchikeys)[train_idx])
    te = set(np.asarray(inchikeys)[test_idx])
    overlap = tr.intersection(te)
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} InChIKey overlaps between train and test.")


def assert_no_scaffold_overlap(train_idx, test_idx, scaffolds) -> None:
    tr = set(np.asarray(scaffolds)[train_idx])
    te = set(np.asarray(scaffolds)[test_idx])
    overlap = tr.intersection(te)
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} scaffold overlaps between train and test.")


def scaffold_stratified_kfold_indices(scaffolds, y, n_splits: int = 5):
    scaffolds = np.asarray(scaffolds)
    y = np.asarray(y).astype(int)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    scaf_to_idx = defaultdict(list)
    for i, scaffold in enumerate(scaffolds):
        scaf_to_idx[scaffold].append(i)

    scaf_items = []
    for scaffold, idxs in scaf_to_idx.items():
        idxs = np.asarray(idxs, dtype=int)
        pos = int(y[idxs].sum())
        size = int(len(idxs))
        scaf_items.append((scaffold, idxs, pos, size))

    scaf_items.sort(key=lambda x: x[3], reverse=True)
    folds = [[] for _ in range(n_splits)]
    fold_pos = np.zeros(n_splits, dtype=int)
    fold_total = np.zeros(n_splits, dtype=int)
    target_ratio = float(y.mean()) if len(y) else 0.0

    for _, idxs, pos, size in scaf_items:
        best_k, best_score = None, None
        for k in range(n_splits):
            new_pos = fold_pos[k] + pos
            new_total = fold_total[k] + size
            new_ratio = new_pos / max(new_total, 1)
            ratio_penalty = abs(new_ratio - target_ratio)
            size_penalty = new_total / max(len(y), 1)
            score = ratio_penalty + 0.25 * size_penalty
            if best_score is None or score < best_score:
                best_score = score
                best_k = k
        folds[best_k].extend(idxs.tolist())
        fold_pos[best_k] += pos
        fold_total[best_k] += size

    all_idx = set(range(len(y)))
    for k in range(n_splits):
        test_idx = np.asarray(sorted(folds[k]), dtype=int)
        train_idx = np.asarray(sorted(list(all_idx - set(test_idx))), dtype=int)
        if len(np.unique(y[test_idx])) < 2:
            # Still yield; safe metrics will handle single-class edge cases.
            pass
        yield train_idx, test_idx
