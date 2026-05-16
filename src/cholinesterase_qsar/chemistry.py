from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdinchi import MolToInchiKey

RDLogger.DisableLog("rdApp.*")


def standardize_mol(mol: Chem.Mol | None, mode: str = "full") -> Chem.Mol | None:
    if mol is None:
        return None
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception:
        rdMolStandardize = None

    try:
        if mode in {"largest_fragment", "uncharged", "full"} and rdMolStandardize is not None:
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        if mode in {"uncharged", "full"} and rdMolStandardize is not None:
            mol = rdMolStandardize.Uncharger().uncharge(mol)
        if mode == "full" and rdMolStandardize is not None:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None


def sanitize_smiles(smiles: str, mode: str = "full") -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        mol = standardize_mol(mol, mode=mode)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def get_scaffold(smiles_std: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return "NONE"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold if scaffold else "NONE"
    except Exception:
        return "NONE"


def to_inchikey(smiles_std: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smiles_std))
        return MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def max_tanimoto_to_train(train_smiles: list[str] | np.ndarray, query_smiles: list[str] | np.ndarray, radius: int, n_bits: int) -> np.ndarray:
    train_bvs = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            train_bvs.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))

    sims_out = []
    for smi in query_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None or not train_bvs:
            sims_out.append(0.0)
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        sims = DataStructs.BulkTanimotoSimilarity(bv, train_bvs)
        sims_out.append(float(max(sims)) if sims else 0.0)
    return np.asarray(sims_out, dtype=float)
