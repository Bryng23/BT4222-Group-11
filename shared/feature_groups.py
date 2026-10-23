"""Feature-group membership for ablations (contract v0.1, "Ablation groups").

Every ablation is a full retrain on the column list returned by
`feature_columns(experiment)`; columns are never dropped at prediction time.
"""
from __future__ import annotations

from . import schema as S

T1 = list(S.T1_DTYPES)
T2 = list(S.T2_DTYPES)
T3_SCALAR = list(S.T3_DTYPES)
U = list(S.U_DTYPES)
H = ["content_smoothed_ctr"]  # built in the pipeline from alpha, not stored

CATEGORICAL = ["category_code", "subcategory_code"]

NEVER_FEATURES = [
    "row_id", "impression_id", "user_id", "news_id", "time", "split", "label",
    "is_zero_exposure", "is_cold_5", "is_cold_10", "is_cold_20", "slate_size",
    *S.BASE_DTYPES,
]

ABLATIONS = {
    "full":                ["T1", "T2", "T3", "U"],
    "no_t1":               ["T2", "T3", "U"],
    "no_t2":               ["T1", "T3", "U"],
    "no_t3":               ["T1", "T2", "U"],
    "t1_only":             ["T1", "U"],
    "t3_only":             ["T3", "U"],
    "full_plus_smoothing": ["T1", "T2", "T3", "U", "H"],
}


def pca_columns(prefix: str, n: int) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(n)]


def group_columns(n_title_pc: int = 32, n_ent_pc: int = 16, include_user_pc: bool = False,
                  include_semantic: bool = True) -> dict[str, list[str]]:
    t2 = [c for c in T2 if include_semantic or c != "semantic_neighbor_ctr"]
    t3 = T3_SCALAR + pca_columns("title_pc", n_title_pc) + pca_columns("ent_pc", n_ent_pc)
    u = list(U)
    if include_user_pc:
        u += pca_columns("user_title_pc", n_title_pc) + pca_columns("user_ent_pc", n_ent_pc)
    return {"T1": list(T1), "T2": t2, "T3": t3, "U": u, "H": list(H)}


def feature_columns(experiment: str, **kwargs) -> list[str]:
    groups = group_columns(**kwargs)
    cols: list[str] = []
    for g in ABLATIONS[experiment]:
        cols += [c for c in groups[g] if c not in cols]
    leaked = set(cols) & set(NEVER_FEATURES)
    assert not leaked, f"never-feature columns in {experiment}: {leaked}"
    return cols
