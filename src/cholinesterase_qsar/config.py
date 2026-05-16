from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class TargetConfig:
    target: str
    train_csv: Path
    bindingdb_tsv: Optional[Path] = None
    pubchem_csv: Optional[Path] = None
    balanced_external_csv: Optional[Path] = None
    external_activity_col: str = "Ki (nM)"
    active_max: float = 100.0
    inactive_min: float = 100000.0
    external_sample_active: int = 100
    external_sample_inactive: int = 75
    remove_external_scaffold_overlap: bool = True
    provisional_external: bool = False
    output_dir: Path = Path("results")
    figure_dir: Path = Path("figures")
    model_dir: Path = Path("models")

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
        return out


@dataclass
class ExperimentConfig:
    random_state: int = 42
    outer_folds: int = 5
    inner_folds_oof: int = 5
    inner_folds_threshold: int = 3
    calibration_cv: int = 3
    n_bits: int = 4096
    radius: int = 3
    use_morgan_counts: bool = True
    morgan_count_bits: int = 2048
    use_rdkitfp: bool = True
    rdkitfp_bits: int = 2048
    use_maccs: bool = True
    use_feature_selection: bool = True
    variance_threshold: float = 0.0
    top_k_features: int = 2500
    ad_similarity_threshold: float = 0.35
    run_preprocess_ablation: bool = True
    run_calibration_analysis: bool = True
    run_external_error_analysis: bool = True
    lgbm_estimators: int = 20000
    lgbm_early_stopping_rounds: int = 300

    def to_dict(self) -> dict:
        return asdict(self)


def target_keywords(target: str, provisional: bool = False) -> list[str]:
    target_norm = target.strip().lower()
    if target_norm == "ache":
        strict = ["acetylcholinesterase"]
        broad = ["acetylcholinesterase", "ache"]
    elif target_norm == "buche":
        strict = ["butyrylcholinesterase", "pseudocholinesterase", "bche"]
        broad = ["butyrylcholinesterase", "pseudocholinesterase", "bche", "cholinesterase"]
    else:
        strict = [target_norm]
        broad = [target_norm]
    return broad if provisional else strict
