"""Independent checks on the metadata and prediction files. Owner: Shi Ting.

Nothing here trusts the feature notebook: cold flags are recomputed from the raw
row log, and every prediction file is checked against meta before it is scored.
Each check returns a DataFrame or dict; the notebook decides what is fatal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema as S

THRESHOLDS = S.COLD_THRESHOLDS


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def check_meta(meta: pd.DataFrame, split: str | None = None) -> dict:
    """Structural checks on one split's metadata table."""
    per_imp = meta.groupby("impression_id")["label"].agg(["size", "sum"])
    res = {
        "rows": len(meta),
        "impressions": len(per_imp),
        "duplicate_row_id": int(meta["row_id"].duplicated().sum()),
        "duplicate_impression_news": int(meta.duplicated(["impression_id", "news_id"]).sum())
        if "news_id" in meta.columns else None,
        "impressions_without_click": int((per_imp["sum"] == 0).sum()),
        "impressions_all_clicked": int((per_imp["sum"] == per_imp["size"]).sum()),
        "slate_size_min": int(per_imp["size"].min()),
        "slate_size_max": int(per_imp["size"].max()),
        "labels_not_binary": int((~meta["label"].isin([0, 1])).sum()),
    }
    if "slate_size" in meta.columns:
        res["slate_size_mismatch"] = int((meta["slate_size"].to_numpy()
                                          != meta.groupby("impression_id")["row_id"].transform("size").to_numpy()).sum())
    if "time" in meta.columns and split is not None:
        t = pd.to_datetime(meta["time"])
        lo = {"train": None, "val": S.VAL_START, "test": S.TEST_START}[split]
        hi = {"train": S.VAL_START, "val": S.TEST_START, "test": None}[split]
        outside = np.zeros(len(t), dtype=bool)
        if lo is not None:
            outside |= (t < lo).to_numpy()
        if hi is not None:
            outside |= (t >= hi).to_numpy()
        res["rows_outside_split_window"] = int(outside.sum())
    return res


def recompute_prior_exposures(meta_all: pd.DataFrame) -> np.ndarray:
    """Exposures of the same article in strictly earlier timestamps, over all splits.

    meta_all is the concatenation of every split (train, val, test). Rows at the same
    timestamp as the current impression do not count, so simultaneous impressions do
    not see each other. Returned in meta_all row order.
    """
    d = pd.DataFrame({"news_id": meta_all["news_id"].to_numpy(), "time": pd.to_datetime(meta_all["time"]).to_numpy(),
                      "pos": np.arange(len(meta_all))}).sort_values(["news_id", "time"], kind="stable")
    d["n_before"] = d.groupby("news_id").cumcount()
    d["n_before"] = d.groupby(["news_id", "time"])["n_before"].transform("min")
    out = np.empty(len(d), dtype=np.int64)
    out[d["pos"].to_numpy()] = d["n_before"].to_numpy()
    return out


def check_cold_flags(meta_all: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Compare stored prior_exposures and is_cold_* flags with an independent recomputation.

    Reports, per split, the share of rows that agree, and how stored flags relate
    to prior_exposures < t (which the evaluator assumes when a flag is missing).
    """
    recomputed = recompute_prior_exposures(meta_all)
    stored = meta_all["prior_exposures"].to_numpy()
    rows = []
    for split, idx in meta_all.groupby("split").indices.items():
        diff = stored[idx] - recomputed[idx]
        row = {"split": split, "rows": len(idx), "prior_exposures_agree": float(np.mean(diff == 0)),
               "median_diff": float(np.median(diff)), "max_abs_diff": int(np.abs(diff).max())}
        for t in thresholds:
            col = f"is_cold_{t}"
            if col in meta_all.columns:
                flag = meta_all[col].to_numpy()[idx].astype(bool)
                row[f"{col}_vs_stored_lt"] = float(np.mean(flag == (stored[idx] < t)))
                row[f"{col}_vs_recomputed_lt"] = float(np.mean(flag == (recomputed[idx] < t)))
        if "is_zero_exposure" in meta_all.columns:
            row["is_zero_vs_stored"] = float(np.mean(meta_all["is_zero_exposure"].to_numpy()[idx] == (stored[idx] == 0)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def check_predictions(meta: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """row_id alignment, duplicates, finiteness and tie structure of one prediction file."""
    m_ids = meta["row_id"].to_numpy()
    p_ids = pred["row_id"].to_numpy()
    s = pred["predicted_score"].to_numpy(np.float64)
    res = {"rows": len(pred), "meta_rows": len(meta),
           "duplicate_row_id": int(pd.Series(p_ids).duplicated().sum()),
           "missing_row_id": int((~np.isin(m_ids, p_ids)).sum()),
           "extra_row_id": int((~np.isin(p_ids, m_ids)).sum()),
           "non_finite": int((~np.isfinite(s)).sum())}
    if res["missing_row_id"] == 0 and res["non_finite"] == 0:
        aligned = meta[["row_id", "impression_id"]].merge(pred[["row_id", "predicted_score"]], on="row_id", how="left")
        nun = aligned.groupby("impression_id")["predicted_score"].agg(["nunique", "size"])
        res["impressions_all_tied"] = float((nun["nunique"] == 1).mean())
        res["share_rows_in_ties"] = float(1 - nun["nunique"].sum() / nun["size"].sum())
    res["ok"] = all(res[k] == 0 for k in ("duplicate_row_id", "missing_row_id", "extra_row_id", "non_finite"))
    return res


def align_scores(meta: pd.DataFrame, pred: pd.DataFrame) -> np.ndarray:
    """Scores in meta row order, joined on row_id (never by position)."""
    sc = meta[["row_id"]].merge(pred[["row_id", "predicted_score"]], on="row_id", how="left", validate="1:1")
    if sc["predicted_score"].isna().any():
        raise ValueError(f"{int(sc['predicted_score'].isna().sum())} meta rows have no prediction")
    return sc["predicted_score"].to_numpy(np.float64)


# ---------------------------------------------------------------------------
# Data support (step 0)
# ---------------------------------------------------------------------------
def cold_support(meta_by_split: dict, thresholds=THRESHOLDS, sd: float = 0.4) -> pd.DataFrame:
    """Cold-click support per split and threshold, before any model is read.

    ci_halfwidth_approx = 1.96 * sd / sqrt(cold-click impressions) is the rough 95%
    interval half-width for a mean Cold Recall (sd=0.4 is the brief's assumption;
    the notebook replaces it with the observed SD once a model is scored).
    """
    rows = []
    for split, m in meta_by_split.items():
        y = m["label"].to_numpy()
        imp = m["impression_id"].to_numpy()
        pe = m["prior_exposures"].to_numpy()
        for t in thresholds:
            cold = pe < t if f"is_cold_{t}" not in m.columns else m[f"is_cold_{t}"].to_numpy().astype(bool)
            cold_clicks = y.astype(bool) & cold
            cold_click_imps = len(np.unique(imp[cold_clicks]))
            neg_cold = pd.Series(cold & ~y.astype(bool)).groupby(imp).any()
            pos_cold = pd.Series(cold_clicks).groupby(imp).any()
            rows.append({"split": split, "threshold": t, "candidate_rows": len(m), "impressions": len(np.unique(imp)),
                         "cold_row_share": float(cold.mean()), "clicks": int(y.sum()),
                         "cold_clicks": int(cold_clicks.sum()), "cold_click_share": float(cold_clicks.sum() / max(y.sum(), 1)),
                         "cold_click_impressions": cold_click_imps,
                         "within_cold_impressions": int((pos_cold & neg_cold).sum()),
                         "zero_exposure_clicks": int((y.astype(bool) & (pe == 0)).sum()),
                         "distinct_cold_clicked_articles": int(m.loc[cold_clicks, "news_id"].nunique())
                         if "news_id" in m.columns else None,
                         "ci_halfwidth_approx": 1.96 * sd / np.sqrt(cold_click_imps) if cold_click_imps else np.nan})
    return pd.DataFrame(rows).set_index(["split", "threshold"])


def threshold_volume(meta_all: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Time and volume needed to leave the cold state, per threshold.

    Answers "why 10 and not 5?" in units of time and traffic: how long after its first
    exposure an article reaches t exposures, and how many articles ever do.
    """
    d = meta_all[["news_id", "time"]].assign(time=pd.to_datetime(meta_all["time"])).sort_values(["news_id", "time"])
    d["k"] = d.groupby("news_id").cumcount() + 1                  # this row is the article's k-th exposure
    first = d.groupby("news_id")["time"].min()
    window_end = d["time"].max()
    daily = d.groupby(["news_id", d["time"].dt.date]).size()
    rows = []
    for t in thresholds:
        reach = d[d["k"] == t].set_index("news_id")["time"]
        hours = (reach - first.reindex(reach.index)).dt.total_seconds() / 3600
        # articles whose first exposure leaves at least 24 h of observed window
        can_24h = first[first <= window_end - pd.Timedelta(hours=24)].index
        rows.append({"threshold": t, "articles": len(first), "share_reaching_t": len(reach) / len(first),
                     "median_hours_to_t": float(hours.median()), "p75_hours_to_t": float(hours.quantile(0.75)),
                     "share_reaching_t_within_24h": float((hours.reindex(can_24h) <= 24).mean())})
    out = pd.DataFrame(rows).set_index("threshold")
    out.attrs["median_exposures_per_article_day"] = float(daily.median())
    out.attrs["mean_exposures_per_article_day"] = float(daily.mean())
    return out


def ctr_noise_curve(meta_all: pd.DataFrame, n_grid=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50),
                    min_total: int = 200) -> pd.DataFrame:
    """How well does CTR after n exposures predict the same article's later CTR?

    Measured on prior exposures (the quantity the cold flag uses), not total
    impressions. For articles with at least min_total exposures in the window:
    early CTR = clicks in the first n exposures / n; later CTR = clicks in the
    remaining exposures / their number. Reports mean absolute error and the binomial
    SD sqrt(p(1-p)/n) at the global CTR. Articles with many exposures are a selected
    (popular) group; state that when using the curve.
    """
    d = meta_all[["news_id", "time", "label"]].assign(time=pd.to_datetime(meta_all["time"]))
    d = d.sort_values(["news_id", "time"], kind="stable")
    d["k"] = d.groupby("news_id").cumcount() + 1
    d["cum"] = d.groupby("news_id")["label"].cumsum()
    tot = d.groupby("news_id").agg(n=("label", "size"), clicks=("label", "sum"))
    tot = tot[tot["n"] >= min_total]
    p = float(meta_all["label"].mean())
    rows = []
    for n in n_grid:
        early = d[d["k"] == n].set_index("news_id")["cum"].reindex(tot.index)
        later = (tot["clicks"] - early) / (tot["n"] - n)
        err = (early / n - later).abs()
        rows.append({"n": n, "articles": len(tot), "mae": float(err.mean()), "median_ae": float(err.median()),
                     "binomial_sd": float(np.sqrt(p * (1 - p) / n)), "global_ctr": p})
    out = pd.DataFrame(rows).set_index("n")
    out["mae_relative_to_n1"] = out["mae"] / out["mae"].iloc[0]
    return out
