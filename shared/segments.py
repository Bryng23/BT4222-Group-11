"""Segment definitions and evaluate_by(meta, score, by=...). Owner: Shi Ting.

Two kinds of segment:

* Impression-level (history length, slate length, time of day, day): every
  candidate in an impression shares the label, so the per-impression metrics
  are grouped by it.
* Candidate-level (exposure bucket, category, subcategory): candidates in one
  impression fall in different segments. For a segment g the metric is the
  full-slate Recall@k of clicked candidates in g (the same definition as Cold
  Recall, with "cold" replaced by g), over impressions with at least one clicked
  candidate in g. AUC g-vs-all, slot share and lift over random come with it.

Every output row carries its support: eligible impressions and clicks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluator as E

EXPOSURE_EDGES = [0, 1, 5, 10, 20, 50, np.inf]
EXPOSURE_LABELS = ["0", "1-4", "5-9", "10-19", "20-49", "50+"]
HISTORY_EDGES = [0, 1, 5, 20, 50, np.inf]
HISTORY_LABELS = ["0", "1-4", "5-19", "20-49", "50+"]
SLATE_EDGES = [0, 10, 25, 50, 100, np.inf]
SLATE_LABELS = ["2-9", "10-24", "25-49", "50-99", "100+"]
HOUR_EDGES = [0, 6, 12, 18, 24]
HOUR_LABELS = ["00-05", "06-11", "12-17", "18-23"]


def _bucket(values, edges, labels) -> pd.Categorical:
    return pd.cut(np.asarray(values, dtype=np.float64), edges, right=False, labels=labels)


def exposure_bucket(meta):
    return _bucket(meta["prior_exposures"], EXPOSURE_EDGES, EXPOSURE_LABELS)


def history_bucket(meta):
    return _bucket(meta["history_length"], HISTORY_EDGES, HISTORY_LABELS)


def slate_bucket(meta):
    n = meta["slate_size"] if "slate_size" in meta.columns else meta.groupby("impression_id")["row_id"].transform("size")
    return _bucket(n, SLATE_EDGES, SLATE_LABELS)


def hour_bucket(meta):
    return _bucket(pd.to_datetime(meta["time"]).dt.hour, HOUR_EDGES, HOUR_LABELS)


def day(meta):
    return pd.to_datetime(meta["time"]).dt.strftime("%m-%d").to_numpy()


# name -> (level, function(meta) -> per-row labels). Add a segment by adding a line.
SEGMENTS = {
    "exposure":    ("candidate",  exposure_bucket),
    "category":    ("candidate",  lambda m: m["category"].to_numpy()),
    "subcategory": ("candidate",  lambda m: m["subcategory"].to_numpy()),
    "history":     ("impression", history_bucket),
    "slate":       ("impression", slate_bucket),
    "hour":        ("impression", hour_bucket),
    "day":         ("impression", day),
}


def _labels(meta, by):
    """Resolve `by` to (level, per-row labels, name)."""
    if isinstance(by, str) and by in SEGMENTS:
        level, fn = SEGMENTS[by]
        return level, fn(meta), by
    if isinstance(by, str) and by in meta.columns:
        lab = meta[by]
        level = "impression" if (lab.groupby(meta["impression_id"]).nunique() <= 1).all() else "candidate"
        return level, lab.to_numpy(), by
    raise KeyError(f"unknown segment {by!r}; add it to SEGMENTS or pass a meta column name")


# ---------------------------------------------------------------------------
# Candidate-level segments
# ---------------------------------------------------------------------------
def candidate_segment_long(meta: pd.DataFrame, score, labels, ks=E.KS, slot_k: int = E.SLOT_K,
                           ranked: E.Ranked | None = None) -> pd.DataFrame:
    """One row per (impression, segment) pair where the segment has at least one candidate.

    Columns: impression_id, user_id (when present), segment, n_candidates (slate),
    n_seg_candidates, n_seg_clicks, n_seg_top{k}, recall@k, hit@k, auc_vs_all.
    Recall, hit and AUC are NaN unless the segment has a click in that impression;
    rows without clicks are kept so slot shares count every impression.
    """
    r = ranked if ranked is not None else E.Ranked(meta, score)
    lab = pd.Categorical(labels)
    code_o = r.sorted(lab.codes)
    nb_all, n_neg = r.neg_below(np.ones_like(r.y_o))
    frame = pd.DataFrame({"g": r.group, "c": code_o, "y": r.y_o, "one": 1.0, "nb": r.y_o * nb_all})
    for k in ks:
        frame[f"h{k}"] = r.y_o * (r.rank <= k)
    frame[f"top{slot_k}"] = (r.rank <= slot_k).astype(float)
    frame = frame[frame["c"] >= 0]
    agg = frame.groupby(["g", "c"], sort=True).sum()
    g = agg.index.get_level_values("g").to_numpy()
    out = pd.DataFrame({"impression_id": r.imp_o[r.starts][g],
                        "segment": lab.categories[agg.index.get_level_values("c")],
                        "n_candidates": r.n[g], "n_seg_candidates": agg["one"].to_numpy(),
                        "n_seg_clicks": agg["y"].to_numpy(), f"n_seg_top{slot_k}": agg[f"top{slot_k}"].to_numpy()})
    if "user_id" in meta.columns:
        out.insert(1, "user_id", r.first(meta["user_id"].to_numpy())[g])
    clicked = out["n_seg_clicks"].to_numpy() > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in ks:
            out[f"recall@{k}"] = np.where(clicked, agg[f"h{k}"].to_numpy() / out["n_seg_clicks"], np.nan)
            out[f"hit@{k}"] = np.where(clicked, (agg[f"h{k}"].to_numpy() > 0).astype(float), np.nan)
            out[f"random_recall@{k}"] = np.where(clicked, np.minimum(k, out["n_candidates"]) / out["n_candidates"], np.nan)
        nn = n_neg[g]
        out["auc_vs_all"] = np.where(clicked & (nn > 0), agg["nb"].to_numpy() / (out["n_seg_clicks"] * nn), np.nan)
    return out


def summarise_candidate_segments(long: pd.DataFrame, ks=E.KS, slot_k: int = E.SLOT_K) -> pd.DataFrame:
    rows = []
    for seg, d in long.groupby("segment", sort=False, observed=True):
        clicked = d[d["n_seg_clicks"] > 0]
        row = {"segment": seg, "eligible_impressions": len(clicked), "clicks": int(d["n_seg_clicks"].sum()),
               "candidates": int(d["n_seg_candidates"].sum())}
        for k in ks:
            row[f"recall@{k}"] = clicked[f"recall@{k}"].mean()
            row[f"recall@{k}_lift"] = row[f"recall@{k}"] - clicked[f"random_recall@{k}"].mean()
            row[f"hit@{k}"] = clicked[f"hit@{k}"].mean()
        row["auc_vs_all"] = clicked["auc_vs_all"].mean()
        row[f"slots_top{slot_k}"] = int(d[f"n_seg_top{slot_k}"].sum())
        rows.append(row)
    out = pd.DataFrame(rows).set_index("segment")
    # Exposure parity: share of all top-k slots vs share of all candidates. Uses every impression
    # where the segment appears, so it is computed from the long table including unclicked rows.
    out[f"slot_share@{slot_k}"] = out[f"slots_top{slot_k}"] / out[f"slots_top{slot_k}"].sum()
    out["candidate_share"] = out["candidates"] / out["candidates"].sum()
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def evaluate_by(meta: pd.DataFrame, score, by="exposure", ks=E.KS, cold_threshold: int = 10,
                per_imp: pd.DataFrame | None = None, metrics=None, return_long: bool = False):
    """Metrics per segment, with eligible impressions and clicks on every row.

    by: a name in SEGMENTS, or any meta column. per_imp: reuse an existing
    per-impression table for impression-level segments (must come from the same meta).
    return_long: also return the per-(impression, segment) table, for bootstrap deltas.
    """
    level, labels, name = _labels(meta, by)
    if level == "candidate":
        long = candidate_segment_long(meta, score, labels, ks)
        # keep impressions where the segment appears but has no click, for slot shares
        table = summarise_candidate_segments(long, ks)
        table.index.name = name
        return (table, long) if return_long else table

    r = E.Ranked(meta, score)
    if per_imp is None:
        per_imp = E.per_impression_metrics(meta, score, cold_threshold=cold_threshold, ks=ks, ranked=r)
    seg = pd.Series(r.first(np.asarray(labels, dtype=object)), index=per_imp.index, name=name)
    rows = {}
    for value, d in per_imp.groupby(seg, sort=True, observed=True):
        s = E.summarise(d)
        rows[value] = {k: s[k] for k in (metrics or s)} | {
            "eligible_impressions": s["n_impressions"], "clicks": s["n_clicks"],
            "cold_click_impressions": s.get("n_cold_click_impressions"), "cold_clicks": s.get("n_cold_clicks")}
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = name
    long = per_imp.assign(segment=seg.to_numpy())
    return (table, long) if return_long else table


def segment_deltas(long_base: pd.DataFrame, long_other: pd.DataFrame, metric: str, n_boot: int = 1000,
                   seed: int = 0, cluster: str = "user_id") -> pd.DataFrame:
    """Per-segment paired delta (other - base) with a user-clustered bootstrap interval.

    Works for both long tables from evaluate_by(..., return_long=True); rows are
    matched on (impression_id, segment).
    """
    key = ["impression_id", "segment"]
    keep = key + ([cluster] if cluster in long_base.columns else []) + [metric]
    m = long_base[keep].merge(long_other[key + [metric]], on=key, suffixes=("_a", "_b"))
    rows = []
    for seg, d in m.groupby("segment", sort=False, observed=True):
        a = d.rename(columns={f"{metric}_a": metric})
        b = d.rename(columns={f"{metric}_b": metric})
        res = E.cluster_bootstrap(a, b, [metric], cluster=cluster if cluster in d.columns else None,
                                  n_boot=n_boot, seed=seed).loc[metric]
        rows.append({"segment": seg, "delta": res["estimate"], "ci_low": res["ci_low"],
                     "ci_high": res["ci_high"], "eligible_impressions": int(res["n_impressions"])})
    return pd.DataFrame(rows).set_index("segment")
