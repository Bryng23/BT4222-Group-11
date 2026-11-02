"""LightGBM hybrid ranker: vector reduction, design matrices, training and scoring.

Rules this module enforces (handover B2-B4):
* PCA is fitted on training-accessible articles only (training candidates plus the
  history articles of training impressions) and then frozen for every split.
* Embedding vectors never sit in the row tables: articles and users are reduced
  once, cached by news_id / impression_id, and joined onto rows as float32 columns.
* An article or user with no vector gets zeros in every PC column, so "no vector"
  is not confused with "average vector". The paired count columns (entity_count,
  history_length) let the model tell the two apart.
* Every model is trained from scratch on its own column list.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from . import derived as D
from . import feature_groups as FG
from . import schema as S

META_COLS = ["row_id", "impression_id", "news_id", "label", "is_cold_10", "is_zero_exposure"]


# ---------------------------------------------------------------------------
# Training-accessible articles and vector reduction
# ---------------------------------------------------------------------------
def train_histories(raw_dir: str, train_impressions) -> set:
    """News ids in the click histories supplied with training impressions."""
    beh = pd.read_csv(os.path.join(raw_dir, "behaviors.tsv"), sep="\t", header=None, usecols=[0, 3],
                      names=["impression_id", "history"])
    beh = beh[beh["impression_id"].isin(set(train_impressions))]
    return {nid for h in beh["history"].dropna() for nid in h.split()}


class VectorReducer:
    """Train-only PCA for title and entity vectors, applied to articles and user profiles."""

    def __init__(self, n_title: int = 32, n_ent: int = 16, seed: int = 4222):
        self.n_title, self.n_ent, self.seed = n_title, n_ent, seed

    def fit(self, feature_root: str, accessible_ids: set):
        ids, title = S.read_vectors(S.article_vectors_path(feature_root), "title_vec", "news_id")
        _, ent = S.read_vectors(S.article_vectors_path(feature_root), "entity_vec", "news_id")
        acc = np.isin(ids.astype(str), np.array(sorted(accessible_ids), dtype=str))
        has_ent = np.linalg.norm(ent, axis=1) > 0
        self.title_pca = PCA(self.n_title, random_state=self.seed).fit(title[acc]) if self.n_title else None
        self.ent_pca = PCA(self.n_ent, random_state=self.seed).fit(ent[acc & has_ent]) if self.n_ent else None
        self.n_fit_articles = int(acc.sum())
        self.n_fit_entity_articles = int((acc & has_ent).sum())
        self.articles = self._frame("news_id", ids.astype(str), title, ent, "title_pc", "ent_pc")
        return self

    def _project(self, pca, mat, prefix, n):
        if pca is None:
            return pd.DataFrame(index=range(len(mat)))
        out = pca.transform(mat).astype(np.float32)
        out[np.linalg.norm(mat, axis=1) == 0] = 0.0
        return pd.DataFrame(out, columns=FG.pca_columns(prefix, n))

    def _frame(self, key, keys, title, ent, t_prefix, e_prefix):
        df = pd.concat([self._project(self.title_pca, title, t_prefix, self.n_title),
                        self._project(self.ent_pca, ent, e_prefix, self.n_ent)], axis=1)
        df.index = pd.Index(keys, name=key)
        return df

    def users(self, feature_root: str, split: str) -> pd.DataFrame:
        path = S.user_vectors_path(feature_root, split)
        ids, title = S.read_vectors(path, "user_title_vec", "impression_id")
        _, ent = S.read_vectors(path, "user_entity_vec", "impression_id")
        return self._frame("impression_id", ids, title, ent, "user_title_pc", "user_ent_pc")

    def explained_variance(self) -> dict:
        return {"title": float(self.title_pca.explained_variance_ratio_.sum()) if self.title_pca else None,
                "entity": float(self.ent_pca.explained_variance_ratio_.sum()) if self.ent_pca else None}


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------
def load_split(feature_root: str, split: str, reducer: VectorReducer, include_user_pc: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(meta, X_all): X_all holds every stored scalar column plus PCA columns, float32 / int16."""
    meta = pd.read_parquet(S.meta_path(feature_root, split), columns=META_COLS)
    feat = pd.read_parquet(S.features_path(feature_root, split))
    assert np.array_equal(meta["row_id"].to_numpy(), feat["row_id"].to_numpy()), f"{split}: row_id misaligned"
    parts = [feat.drop(columns="row_id").reset_index(drop=True),
             reducer.articles.reindex(meta["news_id"].astype(str)).reset_index(drop=True)]
    if include_user_pc:
        parts.append(reducer.users(feature_root, split).reindex(meta["impression_id"]).reset_index(drop=True))
    x = pd.concat(parts, axis=1)
    pcs = [c for c in x.columns if "_pc_" in c]
    assert not x[pcs].isna().any().any(), f"{split}: missing vectors for some rows"
    x[pcs] = x[pcs].astype(np.float32)
    return meta, x


def design(x_all: pd.DataFrame, columns: list[str], alpha: float | None = None) -> pd.DataFrame:
    """Select a model's columns; add content_smoothed_ctr when the H group is requested."""
    x = x_all[[c for c in columns if c != "content_smoothed_ctr"]].copy()
    if "content_smoothed_ctr" in columns:
        assert alpha is not None, "content_smoothed_ctr needs alpha"
        x["content_smoothed_ctr"] = D.content_smoothed_ctr(x_all, alpha)
    return x[columns]


def impression_groups(meta: pd.DataFrame) -> np.ndarray:
    """Group sizes for LambdaRank. Rows of one impression must be contiguous (they are, by row_id order)."""
    imp = meta["impression_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, imp[1:] != imp[:-1]])
    assert len(starts) == len(np.unique(imp)), "impression rows are not contiguous"
    return np.diff(np.r_[starts, len(imp)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BASE_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

PARAM_GRID = [
    {"num_leaves": 31, "min_data_in_leaf": 100},
    {"num_leaves": 31, "min_data_in_leaf": 500},
    {"num_leaves": 127, "min_data_in_leaf": 100},
    {"num_leaves": 127, "min_data_in_leaf": 500},
]


def param_id(p: dict) -> str:
    return ",".join(f"{k}={p[k]}" for k in ("num_leaves", "min_data_in_leaf"))


def fit(x_tr, y_tr, x_va, y_va, params: dict, *, seed: int = 4222, max_rounds: int = 2000, early_stop: int = 100,
        group_tr=None, group_va=None) -> lgb.Booster:
    """Train with early stopping on validation AUC (binary) or nDCG@10 (lambdarank)."""
    p = {**BASE_PARAMS, **params, "seed": seed, "deterministic": True, "force_row_wise": True}
    if p["objective"] == "lambdarank":
        p.update({"metric": "ndcg", "eval_at": [10]})
    else:
        p.setdefault("metric", "auc")
    cat = [c for c in FG.CATEGORICAL if c in x_tr.columns]
    dtr = lgb.Dataset(x_tr, y_tr, group=group_tr, categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(x_va, y_va, group=group_va, categorical_feature=cat, reference=dtr, free_raw_data=False)
    return lgb.train(p, dtr, num_boost_round=max_rounds, valid_sets=[dva], valid_names=["val"],
                     callbacks=[lgb.early_stopping(early_stop, first_metric_only=True, verbose=False)])


def predict(booster: lgb.Booster, x: pd.DataFrame) -> np.ndarray:
    return booster.predict(x, num_iteration=booster.best_iteration).astype(np.float64)


def importance_by_group(booster: lgb.Booster, groups: dict) -> pd.DataFrame:
    """Share of total split gain per feature group (T1/T2/T3/U/H) and per feature."""
    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    gain = gain / gain.sum()
    owner = {c: g for g, cols in groups.items() for c in cols}
    per_feature = gain.rename("gain_share").to_frame().assign(group=lambda d: d.index.map(owner))
    return per_feature.sort_values("gain_share", ascending=False)
