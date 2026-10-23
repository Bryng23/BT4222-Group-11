"""Feature schema contract v0.1: names, dtypes, constants and file layout.

Single source of truth shared by the feature notebook (Andrew), the model
pipeline (Bryan) and the evaluator (Shi Ting). Any change here bumps
SCHEMA_VERSION and must be announced to all three.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "0.1"

# ---- Constants (see contract, "Types, missing values, constants") ----------
K_CONF = 10            # behavior_confidence = n / (n + K_CONF)
M_WIN = 10             # pseudo-exposures shrinking windowed rates toward g_t
M_REL = 50             # pseudo-exposures in the related_ctr hierarchy
WINDOW_HOURS = 6       # recent [t-6h, t), preceding [t-12h, t-6h)
K_NN = 20              # neighbours for semantic_neighbor_ctr
RARE_MIN_ARTICLES = 20 # categories with fewer training-accessible articles -> __rare__
COLD_THRESHOLDS = (5, 10, 20)
PRIMARY_COLD = 10
ALPHA_GRID = (5, 10, 20, 50, 100)

VAL_START = pd.Timestamp("2019-11-13 00:00:00")
TEST_START = pd.Timestamp("2019-11-14 00:00:00")
SPLITS = ("train", "val", "test")

TITLE_DIM = 384
ENTITY_DIM = 100
UNKNOWN_CODE = 0
RARE_CODE = 1

MIND_TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

# ---- Column dtypes ---------------------------------------------------------
META_DTYPES = {
    "row_id": "int64",
    "impression_id": "int32",
    "user_id": "string",
    "news_id": "string",
    "time": "datetime64[ns]",
    "split": "string",
    "label": "int8",
    "slate_size": "int16",
    "prior_exposures": "int32",
    "is_zero_exposure": "bool",
    "is_cold_5": "bool",
    "is_cold_10": "bool",
    "is_cold_20": "bool",
    "history_length": "int16",
    "category": "string",
    "subcategory": "string",
}

T1_DTYPES = {
    "article_age_hours": "float32",
    "prior_exposures": "int32",
    "prior_clicks": "int32",
    "prior_ctr": "float32",
    "behavior_confidence": "float32",
    "recent_exposures_6h": "int32",
    "recent_clicks_6h": "int32",
    "ctr_momentum_6h": "float32",
}
T2_DTYPES = {
    "category_recent_ctr_6h": "float32",
    "category_ctr_momentum_6h": "float32",
    "subcategory_recent_ctr_6h": "float32",
    "subcategory_ctr_momentum_6h": "float32",
    "entity_recent_ctr_mean_6h": "float32",
    "entity_ctr_momentum_mean_6h": "float32",
    "entity_ctr_momentum_max_6h": "float32",
    "semantic_neighbor_ctr": "float32",  # optional
}
T3_DTYPES = {
    "category_code": "int16",
    "subcategory_code": "int16",
    "entity_count": "int16",
    "entity_embedding_coverage": "float32",
}
U_DTYPES = {
    "history_length": "int16",
    "category_affinity": "float32",
    "subcategory_affinity": "float32",
    "text_similarity": "float32",
    "entity_similarity": "float32",
    "entity_overlap_ratio": "float32",
}
BASE_DTYPES = {
    "global_prior_clicks": "int64",
    "global_prior_exposures": "int64",
    "cat_prior_clicks_excl": "int32",
    "cat_prior_exposures_excl": "int32",
    "sub_prior_clicks_excl": "int32",
    "sub_prior_exposures_excl": "int32",
    "global_prior_ctr": "float32",
    "category_prior_ctr": "float32",
    "related_ctr": "float32",
}
OPTIONAL_FEATURES = {"semantic_neighbor_ctr"}

FEATURE_DTYPES = {"row_id": "int64"}
for _d in (T1_DTYPES, T2_DTYPES, T3_DTYPES, U_DTYPES, BASE_DTYPES):
    FEATURE_DTYPES.update(_d)

# Columns that must never exist anywhere: candidate order is shuffled in MIND.
BANNED_COLUMNS = {"position", "pos", "cand_idx", "slate_index", "display_rank", "rank_in_slate"}


# ---- File layout ------------------------------------------------------------
def meta_path(root: str, split: str) -> str:
    return os.path.join(root, f"meta_{split}.parquet")


def features_path(root: str, split: str) -> str:
    return os.path.join(root, f"features_{split}.parquet")


def user_vectors_path(root: str, split: str) -> str:
    return os.path.join(root, f"user_vectors_{split}.parquet")


def article_vectors_path(root: str) -> str:
    return os.path.join(root, "article_vectors.parquet")


def version_path(root: str) -> str:
    return os.path.join(root, "schema_version.json")


def read_version(root: str) -> dict:
    with open(version_path(root)) as f:
        return json.load(f)


# ---- IO helpers --------------------------------------------------------------
def cast(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Return df restricted to the contract columns, in contract order, with contract dtypes."""
    cols = [c for c in dtypes if c in df.columns]
    return df[cols].astype({c: dtypes[c] for c in cols})


def vectors_to_table(key_name: str, keys, vectors: dict, extra: dict | None = None) -> pa.Table:
    """Build an Arrow table with fixed-size float32 list columns from [n, d] arrays."""
    arrays = {key_name: pa.array(keys)}
    for name, mat in vectors.items():
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        arrays[name] = pa.FixedSizeListArray.from_arrays(pa.array(mat.ravel()), mat.shape[1])
    for name, values in (extra or {}).items():
        arrays[name] = pa.array(values)
    return pa.table(arrays)


def read_vectors(path: str, column: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one fixed-size list column back as (keys, [n, d] float32 matrix)."""
    table = pq.read_table(path, columns=[key, column])
    col = table.column(column).combine_chunks()
    dim = col.type.list_size
    mat = col.values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32, copy=False)
    return table.column(key).to_numpy(), mat
