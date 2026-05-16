from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .chemistry import get_scaffold, sanitize_smiles, to_inchikey
from .config import TargetConfig, target_keywords
from .utils import map_class, to_float


def load_internal_dataset(csv_path: Path, standardize_mode: str = "full") -> tuple[pd.DataFrame, dict]:
    df0 = pd.read_csv(csv_path)
    required_cols = ["canonical_smiles", "class"]
    for col in required_cols:
        if col not in df0.columns:
            raise ValueError(f"Required column '{col}' not found in internal dataset: {csv_path}")

    df = df0.copy()
    df["is_active"] = df["class"].map(map_class).astype(int)
    raw_n = len(df)

    df["smiles_std"] = df["canonical_smiles"].map(lambda s: sanitize_smiles(s, mode=standardize_mode))
    n_after_std = int(df["smiles_std"].notna().sum())
    df = df.dropna(subset=["smiles_std"]).reset_index(drop=True)

    df["inchikey"] = df["smiles_std"].map(to_inchikey)
    df = df.dropna(subset=["inchikey"]).reset_index(drop=True)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    df["scaffold"] = df["smiles_std"].map(get_scaffold)

    stats = {
        "raw_n": raw_n,
        "after_std_n": n_after_std,
        "before_dedup_n": n_before_dedup,
        "after_dedup_n": len(df),
        "n_active": int((df["is_active"] == 1).sum()),
        "n_inactive": int((df["is_active"] == 0).sum()),
        "positive_rate": float(df["is_active"].mean()) if len(df) else np.nan,
        "n_unique_scaffolds": int(df["scaffold"].nunique()),
    }
    return df, stats


def quantify_external_contamination(ext_df: pd.DataFrame) -> dict:
    target_col = "Target Name" if "Target Name" in ext_df.columns else None
    if target_col is None:
        return {
            "explicit_buche": 0,
            "generic_cholinesterase_only": 0,
            "ache_or_other_contaminants": 0,
            "unknown_target_text": len(ext_df),
            "explicit_buche_fraction": np.nan,
            "generic_fraction": np.nan,
            "contaminant_fraction": np.nan,
        }

    target_text = ext_df[target_col].astype(str).str.lower()
    explicit_buche = target_text.str.contains("butyrylcholinesterase|pseudocholinesterase|bche", regex=True, na=False)
    explicit_ache = target_text.str.contains("acetylcholinesterase|ache", regex=True, na=False)
    generic_only = target_text.str.contains("cholinesterase", regex=True, na=False) & (~explicit_buche) & (~explicit_ache)
    unknown = ~(explicit_buche | explicit_ache | generic_only)
    contaminants = explicit_ache | unknown
    n = len(ext_df)
    return {
        "explicit_buche": int(explicit_buche.sum()),
        "generic_cholinesterase_only": int(generic_only.sum()),
        "ache_or_other_contaminants": int(contaminants.sum()),
        "unknown_target_text": int(unknown.sum()),
        "explicit_buche_fraction": float(explicit_buche.mean()) if n else np.nan,
        "generic_fraction": float(generic_only.mean()) if n else np.nan,
        "contaminant_fraction": float(contaminants.mean()) if n else np.nan,
    }


def load_bindingdb_external_dataset(cfg: TargetConfig, train_inchikeys, train_scaffolds, standardize_mode: str = "full") -> tuple[pd.DataFrame, dict]:
    if cfg.bindingdb_tsv is None:
        raise ValueError("No BindingDB TSV path was provided.")
    ext = pd.read_csv(cfg.bindingdb_tsv, sep="\t", low_memory=False)
    if "Ligand SMILES" not in ext.columns:
        raise ValueError("BindingDB file must contain 'Ligand SMILES'.")
    if cfg.external_activity_col not in ext.columns:
        raise ValueError(f"BindingDB file must contain '{cfg.external_activity_col}'.")

    if "Target Name" in ext.columns:
        target_text = ext["Target Name"].astype(str).str.lower()
        keep = np.zeros(len(ext), dtype=bool)
        for kw in target_keywords(cfg.target, provisional=cfg.provisional_external):
            keep |= target_text.str.contains(kw, regex=False, na=False)
        ext = ext[keep].copy()

    contamination_pre = quantify_external_contamination(ext)
    ext["potency_nM"] = ext[cfg.external_activity_col].map(to_float)
    ext = ext.dropna(subset=["Ligand SMILES", "potency_nM"]).copy()
    ext["y_ext"] = np.where(ext["potency_nM"] <= cfg.active_max, 1, np.where(ext["potency_nM"] >= cfg.inactive_min, 0, np.nan))
    ext = ext.dropna(subset=["y_ext"]).copy()
    ext["y_ext"] = ext["y_ext"].astype(int)

    ext["smiles_std"] = ext["Ligand SMILES"].map(lambda s: sanitize_smiles(s, mode=standardize_mode))
    ext = ext.dropna(subset=["smiles_std"]).copy()
    ext["inchikey"] = ext["smiles_std"].map(to_inchikey)
    ext = ext.dropna(subset=["inchikey"]).copy()
    ext["scaffold"] = ext["smiles_std"].map(get_scaffold)

    before = len(ext)
    ext = ext[~ext["inchikey"].isin(set(train_inchikeys))].copy()
    after_inchi = len(ext)
    scaffold_overlap_count = int(ext["scaffold"].isin(set(train_scaffolds)).sum())
    if cfg.remove_external_scaffold_overlap:
        ext = ext[~ext["scaffold"].isin(set(train_scaffolds))].copy()
    after_scaf = len(ext)

    act = ext[ext["y_ext"] == 1].copy()
    ina = ext[ext["y_ext"] == 0].copy()
    n_a = min(cfg.external_sample_active, len(act))
    n_i = min(cfg.external_sample_inactive, len(ina))
    act_s = act.sample(n=n_a, random_state=42) if n_a > 0 else act
    ina_s = ina.sample(n=n_i, random_state=42) if n_i > 0 else ina
    ext_s = pd.concat([act_s, ina_s], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    contamination_post = quantify_external_contamination(ext_s)

    stats = {
        "rows_after_target_filter": contamination_pre,
        "n_before_overlap_removal": before,
        "n_after_inchikey_removal": after_inchi,
        "n_after_scaffold_policy": after_scaf,
        "scaffold_overlap_count_after_inchi_filter": scaffold_overlap_count,
        "sampled_n": len(ext_s),
        "sampled_active": int((ext_s["y_ext"] == 1).sum()),
        "sampled_inactive": int((ext_s["y_ext"] == 0).sum()),
        "sampled_contamination": contamination_post,
    }
    return ext_s, stats


def load_pubchem_external_dataset(csv_path: Path, cfg: TargetConfig, train_inchikeys, train_scaffolds, standardize_mode: str = "full") -> tuple[pd.DataFrame, dict]:
    ext0 = pd.read_csv(csv_path, low_memory=False)
    required_cols = ["canonical_smiles", "target", "activity_type", "activity_value_nM"]
    for col in required_cols:
        if col not in ext0.columns:
            raise ValueError(f"Required column '{col}' not found in PubChem external CSV.")
    ext = ext0.copy()
    ext = ext[
        ext["target"].astype(str).str.strip().str.lower().eq(cfg.target.lower())
        & ext["activity_type"].astype(str).str.strip().str.upper().eq("IC50")
    ].copy()
    ext["potency_nM"] = ext["activity_value_nM"].map(to_float)
    ext = ext.dropna(subset=["canonical_smiles", "potency_nM"]).copy()
    ext["y_ext"] = np.where(ext["potency_nM"] <= cfg.active_max, 1, np.where(ext["potency_nM"] >= cfg.inactive_min, 0, np.nan))
    ext = ext.dropna(subset=["y_ext"]).copy()
    ext["y_ext"] = ext["y_ext"].astype(int)
    ext["smiles_std"] = ext["canonical_smiles"].map(lambda s: sanitize_smiles(s, mode=standardize_mode))
    ext = ext.dropna(subset=["smiles_std"]).copy()
    ext["inchikey"] = ext["smiles_std"].map(to_inchikey)
    ext = ext.dropna(subset=["inchikey"]).copy()
    ext["scaffold"] = ext["smiles_std"].map(get_scaffold)
    before = len(ext)
    ext = ext[~ext["inchikey"].isin(set(train_inchikeys))].copy()
    after_inchi = len(ext)
    scaffold_overlap_count = int(ext["scaffold"].isin(set(train_scaffolds)).sum())
    before_dedup = len(ext)
    ext = ext.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    stats = {
        "raw_rows": len(ext0),
        "n_before_overlap_removal": before,
        "n_after_inchikey_removal": after_inchi,
        "n_before_external_dedup": before_dedup,
        "final_n": len(ext),
        "n_active": int((ext["y_ext"] == 1).sum()),
        "n_inactive": int((ext["y_ext"] == 0).sum()),
        "positive_rate": float(ext["y_ext"].mean()) if len(ext) else np.nan,
        "scaffold_overlap_count": scaffold_overlap_count,
        "median_potency_nM": float(ext["potency_nM"].median()) if len(ext) else np.nan,
    }
    return ext, stats


def load_labeled_external_dataset(csv_path: Path, train_inchikeys, train_scaffolds, standardize_mode: str = "full") -> tuple[pd.DataFrame, dict]:
    ext0 = pd.read_csv(csv_path, low_memory=False)
    required_cols = ["smiles", "class"]
    for col in required_cols:
        if col not in ext0.columns:
            raise ValueError(f"Required column '{col}' not found in labeled external CSV.")
    ext = ext0.copy()
    ext["class"] = ext["class"].astype(str).str.strip().str.lower()
    valid = ext["class"].isin(["active", "inactive"])
    ext = ext[valid].copy()
    ext["y_ext"] = ext["class"].map({"inactive": 0, "active": 1}).astype(int)
    ext["smiles_std"] = ext["smiles"].map(lambda s: sanitize_smiles(s, mode=standardize_mode))
    ext = ext.dropna(subset=["smiles_std"]).copy()
    ext["inchikey"] = ext["smiles_std"].map(to_inchikey)
    ext = ext.dropna(subset=["inchikey"]).copy()
    ext["scaffold"] = ext["smiles_std"].map(get_scaffold)
    before = len(ext)
    ext = ext[~ext["inchikey"].isin(set(train_inchikeys))].copy()
    after_inchi = len(ext)
    scaffold_overlap_count = int(ext["scaffold"].isin(set(train_scaffolds)).sum())
    before_dedup = len(ext)
    ext = ext.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    stats = {
        "raw_rows": len(ext0),
        "rows_with_valid_class": int(valid.sum()),
        "n_before_overlap_removal": before,
        "n_after_inchikey_removal": after_inchi,
        "n_before_external_dedup": before_dedup,
        "final_n": len(ext),
        "n_active": int((ext["y_ext"] == 1).sum()),
        "n_inactive": int((ext["y_ext"] == 0).sum()),
        "positive_rate": float(ext["y_ext"].mean()) if len(ext) else np.nan,
        "scaffold_overlap_count": scaffold_overlap_count,
    }
    return ext, stats
