from __future__ import annotations

import argparse
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cholinesterase_qsar.config import ExperimentConfig, TargetConfig
from cholinesterase_qsar.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AChE/BuChE ensemble QSAR pipeline.")
    parser.add_argument("--target", required=True, choices=["AChE", "BuChE"], help="Target name.")
    parser.add_argument("--train-csv", required=True, type=Path, help="Internal training CSV path.")
    parser.add_argument("--bindingdb-tsv", type=Path, default=None, help="Optional BindingDB-style external TSV path.")
    parser.add_argument("--pubchem-csv", type=Path, default=None, help="Optional PubChem-style external CSV path.")
    parser.add_argument("--balanced-external-csv", type=Path, default=None, help="Optional balanced labeled external CSV path.")
    parser.add_argument("--external-activity-col", default="Ki (nM)", help="External BindingDB potency column name.")
    parser.add_argument("--active-max", type=float, default=None, help="Activity threshold for active class in nM.")
    parser.add_argument("--inactive-min", type=float, default=None, help="Activity threshold for inactive class in nM.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/run"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/run"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/run"))
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds-oof", type=int, default=5)
    parser.add_argument("--inner-folds-threshold", type=int, default=3)
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument("--top-k-features", type=int, default=2500)
    parser.add_argument("--variance-threshold", type=float, default=0.0)
    parser.add_argument("--ad-sim-threshold", type=float, default=0.35)
    parser.add_argument("--lgbm-estimators", type=int, default=20000)
    parser.add_argument("--lgbm-early-stopping-rounds", type=int, default=300)
    parser.add_argument("--skip-preprocess-ablation", action="store_true")
    parser.add_argument("--skip-calibration-analysis", action="store_true")
    parser.add_argument("--no-feature-selection", action="store_true")
    parser.add_argument("--remove-external-scaffold-overlap", action="store_true")
    parser.add_argument("--provisional-external", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.active_max is None:
        active_max = 100.0 if args.target == "AChE" else 1000.0
    else:
        active_max = args.active_max

    if args.inactive_min is None:
        inactive_min = 100000.0 if args.target == "AChE" else 10000.0
    else:
        inactive_min = args.inactive_min

    remove_scaffold_overlap = args.remove_external_scaffold_overlap
    if args.target == "AChE" and not args.provisional_external:
        remove_scaffold_overlap = True

    target_cfg = TargetConfig(
        target=args.target,
        train_csv=args.train_csv,
        bindingdb_tsv=args.bindingdb_tsv,
        pubchem_csv=args.pubchem_csv,
        balanced_external_csv=args.balanced_external_csv,
        external_activity_col=args.external_activity_col,
        active_max=active_max,
        inactive_min=inactive_min,
        remove_external_scaffold_overlap=remove_scaffold_overlap,
        provisional_external=args.provisional_external,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        model_dir=args.model_dir,
    )

    exp_cfg = ExperimentConfig(
        outer_folds=args.outer_folds,
        inner_folds_oof=args.inner_folds_oof,
        inner_folds_threshold=args.inner_folds_threshold,
        calibration_cv=args.calibration_cv,
        use_feature_selection=not args.no_feature_selection,
        variance_threshold=args.variance_threshold,
        top_k_features=args.top_k_features,
        ad_similarity_threshold=args.ad_sim_threshold,
        run_preprocess_ablation=not args.skip_preprocess_ablation,
        run_calibration_analysis=not args.skip_calibration_analysis,
        lgbm_estimators=args.lgbm_estimators,
        lgbm_early_stopping_rounds=args.lgbm_early_stopping_rounds,
    )

    run_pipeline(target_cfg, exp_cfg)


if __name__ == "__main__":
    main()
