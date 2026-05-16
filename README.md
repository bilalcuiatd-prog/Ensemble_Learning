# Cholinesterase QSAR Ensemble Learning Pipeline

A reproducible machine-learning pipeline for binary QSAR classification of acetylcholinesterase (AChE) and butyrylcholinesterase (BuChE) inhibitors using leakage-safe molecular curation, fingerprint engineering, scaffold-stratified nested cross-validation, stacked ensemble learning, calibration, applicability-domain analysis, and external validation.

## Key features

- AChE and BuChE target-specific configuration
- RDKit molecular standardization and InChIKey-based deduplication
- Bemis-Murcko scaffold extraction
- ECFP6 bit fingerprints, ECFP6 count fingerprints, RDKit fingerprints, MACCS keys, and physicochemical descriptors
- Train-only variance filtering and mutual-information feature selection
- Nested scaffold cross-validation
- Base learners: LightGBM, Random Forest, Extra Trees, and SVD-Logistic Regression
- Meta-learners: Logistic Regression, Random Forest, and LightGBM
- Optional sigmoid calibration
- Threshold optimization for balanced accuracy and accuracy
- External evaluation from BindingDB-style TSV, PubChem-style CSV, and balanced labeled CSV files
- Applicability-domain reporting using maximum Tanimoto similarity
- Publication-ready tables and figures

## Repository structure

```text
.
├── data/
│   ├── raw/                  # Internal AChE/BuChE training CSV files
│   └── external/             # External BindingDB/PubChem/balanced CSV files
├── figures/                  # Generated figures
├── models/                   # Saved trained models
├── results/                  # Metrics, predictions, and tables
├── scripts/
│   ├── run_pipeline.py       # Main full pipeline runner
│   └── run_external_only.py  # Optional external-only evaluation after training artifact exists
├── src/
│   └── cholinesterase_qsar/
│       ├── __init__.py
│       ├── config.py
│       ├── utils.py
│       ├── chemistry.py
│       ├── features.py
│       ├── data.py
│       ├── splitting.py
│       ├── metrics.py
│       ├── models.py
│       ├── plots.py
│       ├── external.py
│       └── pipeline.py
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── .gitignore
```

## Input file requirements

### Internal training CSV

Your internal training file should contain at least:

```text
canonical_smiles,class
```

The `class` column should contain values such as:

```text
active, inactive
```

or binary equivalents such as `1` and `0`.

### BindingDB-style external TSV

The default loader expects:

```text
Ligand SMILES
Target Name
Ki (nM)
```

The activity column can be changed with `--external-activity-col`.

### PubChem-style external CSV

The PubChem loader expects:

```text
canonical_smiles,target,activity_type,activity_value_nM
```

### Balanced labeled external CSV

The labeled external loader expects:

```text
smiles,class
```

where `class` is `active` or `inactive`.

## Installation

### Recommended: conda

```bash
conda env create -f environment.yml
conda activate cholinesterase-qsar
```

### Alternative: pip

```bash
pip install -r requirements.txt
```

RDKit is most reliable through conda.

## Example: run AChE

```bash
python scripts/run_pipeline.py ^
  --target AChE ^
  --train-csv data/raw/Acetylcholinesterase_5.csv ^
  --bindingdb-tsv data/external/bindingdb_ache.tsv ^
  --balanced-external-csv data/external/AChE_balanced_500.csv ^
  --output-dir results/ache ^
  --figure-dir figures/ache ^
  --model-dir models/ache
```

For Mac/Linux, replace `^` with `\`.

## Example: run BuChE

```bash
python scripts/run_pipeline.py ^
  --target BuChE ^
  --train-csv data/raw/Butyrylcholinesterase_5.csv ^
  --bindingdb-tsv data/external/bindingdb_buche.tsv ^
  --pubchem-csv data/external/BuChE_pubchem_ic50_smiles.csv ^
  --balanced-external-csv data/external/BuChE_balanced_500.csv ^
  --output-dir results/buche ^
  --figure-dir figures/buche ^
  --model-dir models/buche
```

## Faster test run

For a quick check before full training:

```bash
python scripts/run_pipeline.py ^
  --target AChE ^
  --train-csv data/raw/Acetylcholinesterase_5.csv ^
  --output-dir results/test_ache ^
  --figure-dir figures/test_ache ^
  --model-dir models/test_ache ^
  --outer-folds 3 ^
  --inner-folds-oof 3 ^
  --inner-folds-threshold 3 ^
  --top-k-features 1000 ^
  --lgbm-estimators 1000 ^
  --skip-preprocess-ablation
```

## Main outputs

The pipeline saves:

```text
results/<target>/internal_curation_stats.csv
results/<target>/nested_cv_fold_results.csv
results/<target>/nested_cv_summary.csv
results/<target>/runtime_breakdown.csv
results/<target>/meta_learner_comparison.csv
results/<target>/external_*_summary.csv
results/<target>/external_*_predictions.csv
models/<target>/final_artifact.joblib
figures/<target>/*.png
```

## Important methodological notes

- Feature selection is fitted only on the training portion of each outer fold.
- Scaffold groups are not shared across train/test folds.
- InChIKey overlap is explicitly checked to prevent duplicate leakage.
- External datasets are filtered for exact molecular overlap with the internal training set.
- Scaffold overlap in external data is quantified and can be removed when required.

## Citation

If this repository supports a paper, cite the associated article and include the GitHub repository URL in the Data/Code Availability section.
