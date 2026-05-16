from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from .config import ExperimentConfig

DESC_NAMES = [
    "MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "HeavyAtomCount", "RingCount",
    "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings",
    "FractionCSP3", "HallKierAlpha",
]


def bv_to_np(bv) -> np.ndarray:
    arr = np.zeros((bv.GetNumBits(),), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_bits(smiles_std: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return bv_to_np(bv)


def morgan_counts(smiles_std: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    for key, value in fp.GetNonzeroElements().items():
        arr[int(key)] = float(value)
    return np.log1p(arr).astype(np.float32)


def rdkitfp_bits(smiles_std: str, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = Chem.RDKFingerprint(mol, fpSize=n_bits)
    return bv_to_np(bv)


def maccs_bits(smiles_std: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    bv = MACCSkeys.GenMACCSKeys(mol)
    return bv_to_np(bv)


def physchem_desc_13(smiles_std: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles_std))
    if mol is None:
        return np.zeros(13, dtype=np.float32)
    return np.asarray([
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HallKierAlpha(mol),
    ], dtype=np.float32)


def build_feature_matrix(smiles_std_list: list[str] | np.ndarray, exp: ExperimentConfig) -> tuple[np.ndarray, list[str]]:
    fps_bits, fps_counts, fps_rd, fps_maccs, descs = [], [], [], [], []
    for smi in smiles_std_list:
        fps_bits.append(morgan_bits(smi, exp.radius, exp.n_bits))
        if exp.use_morgan_counts:
            fps_counts.append(morgan_counts(smi, exp.radius, exp.morgan_count_bits))
        if exp.use_rdkitfp:
            fps_rd.append(rdkitfp_bits(smi, exp.rdkitfp_bits))
        if exp.use_maccs:
            fps_maccs.append(maccs_bits(smi))
        descs.append(physchem_desc_13(smi))

    blocks = [np.stack(fps_bits).astype(np.float32)]
    names = [f"ECFP6_bit_{i}" for i in range(exp.n_bits)]

    if exp.use_morgan_counts:
        blocks.append(np.stack(fps_counts).astype(np.float32))
        names += [f"ECFP6_count_{i}" for i in range(exp.morgan_count_bits)]
    if exp.use_rdkitfp:
        blocks.append(np.stack(fps_rd).astype(np.float32))
        names += [f"RDKFP_{i}" for i in range(exp.rdkitfp_bits)]
    if exp.use_maccs:
        blocks.append(np.stack(fps_maccs).astype(np.float32))
        names += [f"MACCS_{i}" for i in range(167)]

    blocks.append(np.stack(descs).astype(np.float32))
    names += DESC_NAMES
    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, names


class TrainOnlyFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variance_threshold: float = 0.0, top_k: int = 2500, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, X, y):
        self.var_filter_ = VarianceThreshold(threshold=self.variance_threshold)
        Xv = self.var_filter_.fit_transform(X)
        k = min(self.top_k, Xv.shape[1])
        mi = mutual_info_classif(Xv, y, discrete_features="auto", random_state=self.random_state)
        idx = np.argsort(mi)[::-1][:k]
        self.mi_indices_ = np.asarray(sorted(idx), dtype=int)
        self.n_input_features_ = X.shape[1]
        self.n_after_variance_ = Xv.shape[1]
        self.n_selected_ = len(self.mi_indices_)
        return self

    def transform(self, X):
        Xv = self.var_filter_.transform(X)
        return Xv[:, self.mi_indices_]
