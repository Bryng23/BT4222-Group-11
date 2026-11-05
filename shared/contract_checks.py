"""Acceptance checks from contract v0.1 for the exported feature tables.

`validate_tables` covers checks 2-6, 9 and 10 on files written to disk.
`leakage_permutation_test` (check 7) and `brute_force_check` (check 8) need the
raw log and a feature-building function, so the feature notebook can call them
with its own builder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S


class ContractError(AssertionError):
    pass


def _require(cond, msg):
    if not cond:
        raise ContractError(msg)


def load_split(root: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(S.meta_path(root, split)), pd.read_parquet(S.features_path(root, split))


def validate_tables(root: str, expected_rows: int | None = None,
                    expected_impressions: dict | None = None) -> dict:
    """Load every split from `root` and assert the contract. Returns a summary dict."""
    metas, feats = [], []
    for split in S.SPLITS:
        m, f = load_split(root, split)
        _require((m["split"] == split).all(), f"{split}: split column mismatch")
        _require(np.array_equal(m["row_id"].to_numpy(), f["row_id"].to_numpy()), f"{split}: meta/features row_id misaligned")
        metas.append(m)
        feats.append(f)
    meta = pd.concat(metas, ignore_index=True)
    feat = pd.concat(feats, ignore_index=True)

    # dtypes and columns
    for col, dt in S.META_DTYPES.items():
        _require(col in meta.columns, f"meta missing column {col}")
        _require(str(meta[col].dtype).startswith(dt.split("[")[0]) or (dt == "string" and meta[col].dtype == object),
                 f"meta.{col} has dtype {meta[col].dtype}, expected {dt}")
    for col, dt in S.FEATURE_DTYPES.items():
        if col in S.OPTIONAL_FEATURES and col not in feat.columns:
            continue
        _require(col in feat.columns, f"features missing column {col}")
        _require(str(feat[col].dtype) == dt, f"features.{col} has dtype {feat[col].dtype}, expected {dt}")
    extra = set(feat.columns) - set(S.FEATURE_DTYPES)
    _require(not extra, f"features has columns outside the contract: {sorted(extra)}")
    banned = (set(meta.columns) | set(feat.columns)) & S.BANNED_COLUMNS
    _require(not banned, f"position-like columns present: {banned}")

    # keys
    _require(np.array_equal(meta["row_id"].to_numpy(), np.arange(len(meta))), "row_id is not contiguous 0..N-1")
    _require(not meta.duplicated(["impression_id", "news_id"]).any(), "(impression_id, news_id) not unique")
    if expected_rows is not None:
        _require(len(meta) == expected_rows, f"row count {len(meta)} != {expected_rows}")
    if expected_impressions:
        got = meta.groupby("split")["impression_id"].nunique().to_dict()
        for s, n in expected_impressions.items():
            _require(got.get(s) == n, f"{s}: {got.get(s)} impressions, expected {n}")

    # split boundaries
    t = meta["time"]
    _require(((t < S.VAL_START) == (meta["split"] == "train")).all(), "train rows outside train window")
    _require(((t >= S.TEST_START) == (meta["split"] == "test")).all(), "test rows outside test window")

    # no NaN / inf
    num = feat.select_dtypes("number")
    bad = num.columns[~np.isfinite(num.to_numpy(np.float64)).all(0)].tolist()
    _require(not bad, f"NaN/inf in {bad}")

    # ranges
    rates = [c for c in feat.columns if c.endswith(("_ctr", "_ctr_6h", "_mean_6h", "_affinity", "_coverage", "_ratio"))
             and "momentum" not in c] + ["behavior_confidence"]
    for c in rates:
        _require(feat[c].between(0, 1).all(), f"{c} outside [0, 1]")
    for c in [c for c in feat.columns if "momentum" in c] + ["text_similarity", "entity_similarity"]:
        _require(feat[c].between(-1 - 1e-5, 1 + 1e-5).all(), f"{c} outside [-1, 1]")
    _require((feat["prior_clicks"] <= feat["prior_exposures"]).all(), "prior_clicks > prior_exposures")
    _require((feat["recent_exposures_6h"] <= feat["prior_exposures"]).all(), "recent exposures > prior exposures")
    _require((feat["recent_clicks_6h"] <= feat["recent_exposures_6h"]).all(), "recent clicks > recent exposures")
    _require((feat["article_age_hours"] >= 0).all(), "negative article age")

    # identities
    e, c = feat["prior_exposures"].to_numpy(np.float64), feat["prior_clicks"].to_numpy(np.float64)
    ctr = np.divide(c, e, out=np.zeros_like(e), where=e > 0)
    _require(np.allclose(feat["prior_ctr"], ctr, atol=1e-6), "prior_ctr != prior_clicks / prior_exposures")
    _require(np.allclose(feat["behavior_confidence"], e / (e + S.K_CONF), atol=1e-6), "behavior_confidence != e/(e+k)")
    _require((meta["prior_exposures"].to_numpy() == feat["prior_exposures"].to_numpy()).all(), "meta/features prior_exposures differ")
    for th in S.COLD_THRESHOLDS:
        _require((meta[f"is_cold_{th}"] == (meta["prior_exposures"] < th)).all(), f"is_cold_{th} inconsistent")
    _require((meta["is_zero_exposure"] == (meta["prior_exposures"] == 0)).all(), "is_zero_exposure inconsistent")
    _require((meta["history_length"].to_numpy() == feat["history_length"].to_numpy()).all(), "history_length differs")

    # monotone counters per article
    order = meta.sort_values(["news_id", "time"], kind="mergesort").index
    d = meta.loc[order].groupby("news_id")["prior_exposures"].diff().dropna()
    _require((d >= 0).all(), "prior_exposures decreases over time for some article")

    return summarise(meta, feat)


def summarise(meta: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """Per-split counts used in the report: cold shares, zero exposure, edge cases."""
    out = {}
    for split, m in meta.groupby("split", sort=False):
        f = feat.loc[m.index]
        out[split] = {
            "impressions": int(m["impression_id"].nunique()),
            "rows": int(len(m)),
            "ctr": round(float(m["label"].mean()), 4),
            "zero_exposure_rows": int(m["is_zero_exposure"].sum()),
            **{f"cold_{th}_share": round(float(m[f"is_cold_{th}"].mean()), 4) for th in S.COLD_THRESHOLDS},
            "empty_history_rows": int((m["history_length"] == 0).sum()),
            "no_entity_rows": int((f["entity_count"] == 0).sum()),
            "entities_without_embedding_rows": int(((f["entity_count"] > 0) & (f["entity_embedding_coverage"] < 1)).sum()),
            "unknown_category_rows": int((f["category_code"] == S.UNKNOWN_CODE).sum()),
            "rare_category_rows": int((f["category_code"] == S.RARE_CODE).sum()),
            "slate_size_min": int(m["slate_size"].min()),
            "slate_size_max": int(m["slate_size"].max()),
        }
    return out


def leakage_permutation_test(build_fn, rows: pd.DataFrame, cutoff, seed: int = 0, **build_kwargs) -> int:
    """Check 7: shuffle labels of rows with time >= cutoff, rebuild, and require every
    feature of rows with time <= cutoff to be unchanged. Returns the number of rows compared.

    build_fn(rows, **build_kwargs) must return a dict with a "meta" and a "features" frame.
    """
    cutoff = pd.Timestamp(cutoff)
    a = build_fn(rows, **build_kwargs)
    shuffled = rows.copy()
    late = shuffled["time"] >= cutoff
    shuffled.loc[late, "label"] = np.random.default_rng(seed).permutation(shuffled.loc[late, "label"].to_numpy())
    b = build_fn(shuffled, **build_kwargs)
    key = ["impression_id", "news_id"]
    fa = a["meta"][key + ["time"]].join(a["features"].drop(columns="row_id"))
    fb = b["meta"][key].join(b["features"].drop(columns="row_id"))
    merged = fa.merge(fb, on=key, suffixes=("_a", "_b"))
    early = merged["time"] <= cutoff
    cols = [c for c in a["features"].columns if c != "row_id"]
    diff = [c for c in cols if not np.allclose(merged.loc[early, f"{c}_a"], merged.loc[early, f"{c}_b"], atol=1e-6)]
    _require(not diff, f"features before {cutoff} changed when later labels were shuffled: {diff}")
    _require(len(merged) == len(rows), "rebuild changed the row set")
    return int(early.sum())


def brute_force_check(meta: pd.DataFrame, feat: pd.DataFrame, rows: pd.DataFrame, news: pd.DataFrame,
                      n_sample: int = 500, seed: int = 0) -> int:
    """Check 8: recompute T1, category windows and related_ctr for sampled rows with
    plain boolean masks over the raw log and compare. Returns the number of rows checked."""
    log = rows.merge(news[["news_id", "category", "subcategory"]], on="news_id")
    joined = meta.merge(feat, on="row_id", suffixes=("", "_f"))
    sample = joined.sample(min(n_sample, len(joined)), random_state=seed)
    w = pd.Timedelta(hours=S.WINDOW_HOURS)
    for _, r in sample.iterrows():
        t = r["time"]
        before = log[log["time"] < t]
        g = before["label"].mean() if len(before) else 0.0
        own = before[before["news_id"] == r["news_id"]]
        _require(len(own) == r["prior_exposures"], f"row {r.row_id}: prior_exposures")
        _require(own["label"].sum() == r["prior_clicks"], f"row {r.row_id}: prior_clicks")
        recent = own[own["time"] >= t - w]
        _require(len(recent) == r["recent_exposures_6h"], f"row {r.row_id}: recent_exposures_6h")
        _require(abs(g - r["global_prior_ctr"]) < 1e-6, f"row {r.row_id}: global_prior_ctr")

        others = before[before["news_id"] != r["news_id"]]
        cat = others[others["category"] == r["category"]]
        now = cat[cat["time"] >= t - w]
        rate = (now["label"].sum() + S.M_WIN * g) / (len(now) + S.M_WIN)
        _require(abs(rate - r["category_recent_ctr_6h"]) < 1e-5, f"row {r.row_id}: category_recent_ctr_6h")

        sub = others[others["subcategory"] == r["subcategory"]]
        r_cat = (cat["label"].sum() + S.M_REL * g) / (len(cat) + S.M_REL)
        r_sub = (sub["label"].sum() + S.M_REL * r_cat) / (len(sub) + S.M_REL)
        expect = g if r["category_code"] == S.UNKNOWN_CODE else (r_cat if r["subcategory_code"] <= S.RARE_CODE else r_sub)
        _require(abs(expect - r["related_ctr"]) < 1e-5, f"row {r.row_id}: related_ctr")
    return len(sample)
