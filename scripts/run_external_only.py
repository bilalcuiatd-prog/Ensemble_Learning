from __future__ import annotations

import argparse
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib

from cholinesterase_qsar.config import ExperimentConfig, TargetConfig
from cholinesterase_qsar.data import load_bindingdb_external_dataset, load_labeled_external_dataset, load_pubchem_external_dataset
from cholinesterase_qsar.external import evaluate_external_dataset
from cholinesterase_qsar.utils import ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external evaluation from a saved training artifact.")
    parser.add_argument("--artifact", required=True, type=Path, help="Path to final_artifact.joblib.")
    parser.add_argument("--external-type", required=True, choices=["bindingdb", "pubchem", "balanced"])
    parser.add_argument("--external-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--external-activity-col", default="Ki (nM)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.artifact)
    target_dict = artifact["target_config"]
    exp_dict = artifact["experiment_config"]
    cfg = TargetConfig(
        target=target_dict["target"],
        train_csv=Path(target_dict["train_csv"]),
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        model_dir=Path(target_dict.get("model_dir", "models")),
        external_activity_col=args.external_activity_col,
        active_max=float(target_dict.get("active_max", 100.0)),
        inactive_min=float(target_dict.get("inactive_min", 100000.0)),
        provisional_external=bool(target_dict.get("provisional_external", False)),
        remove_external_scaffold_overlap=bool(target_dict.get("remove_external_scaffold_overlap", False)),
    )
    exp = ExperimentConfig(**{k: v for k, v in exp_dict.items() if k in ExperimentConfig.__dataclass_fields__})
    ensure_dirs(cfg.output_dir, cfg.figure_dir)

    if args.external_type == "bindingdb":
        cfg.bindingdb_tsv = args.external_file
        ext_df, ext_stats = load_bindingdb_external_dataset(cfg, artifact["internal_inchikeys"], artifact["internal_scaffolds"])
        label = "BindingDB"
    elif args.external_type == "pubchem":
        ext_df, ext_stats = load_pubchem_external_dataset(args.external_file, cfg, artifact["internal_inchikeys"], artifact["internal_scaffolds"])
        label = "PubChem"
    else:
        ext_df, ext_stats = load_labeled_external_dataset(args.external_file, artifact["internal_inchikeys"], artifact["internal_scaffolds"])
        label = "Balanced"

    evaluate_external_dataset(
        ext_df=ext_df,
        ext_stats=ext_stats,
        label=label,
        cfg=cfg,
        exp=exp,
        feature_selector=artifact["feature_selector"],
        trained_full_base=artifact["trained_full_base"],
        final_meta_model=artifact["final_meta_model"],
        base_model_names=artifact["base_model_names"],
        smiles_train=artifact["internal_smiles"],
        threshold_bacc=artifact["thr_bacc_final"],
        threshold_acc=artifact["thr_acc_final"],
    )


if __name__ == "__main__":
    main()
