"""INTERIM evaluator, used only for validation tuning until Shi Ting's evaluator exists.

Local-only: swap the import for her module once it lands, and never report
final numbers from this file. Definitions follow the MIND / Recommenders
conventions so the switch should not change any selection:

* Ranking is per impression by score, descending. Ties are broken by a seeded
  hash of row_id, which ignores labels and is identical for every model.
* AUC: per impression, ties counted as 1/2 (average ranks); impressions with a
  single label class are skipped; mean over impressions.
* MRR: per impression, mean of 1/rank over clicked candidates; mean over impressions.
* nDCG@k: binary gains, log2 discount; mean over impressions.
* Cold Recall@k: clicked cold candidates ranked in the full-slate top k divided by
  clicked cold candidates; mean over impressions with at least one cold click.
  Cold Hit@k: share of those impressions with at least one cold click in the top k.
  Warm Recall@k is the same quantity for clicked warm candidates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_SEED = 4222


def tie_key(row_id, seed: int = TIE_SEED) -> np.ndarray:
    """Deterministic pseudo-random key per row (splitmix64 of row_id + seed)."""
    with np.errstate(over="ignore"):
        z = np.asarray(row_id, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def per_impression_metrics(meta: pd.DataFrame, score, cold_col: str = "is_cold_10",
                           ks=(5, 10), seed: int = TIE_SEED) -> pd.DataFrame:
    """One row per impression with every metric (NaN where the impression is not eligible).

    meta needs row_id, impression_id, label and cold_col, aligned with score.
    """
    imp = meta["impression_id"].to_numpy()
    y = meta["label"].to_numpy().astype(np.int64)
    cold = meta[cold_col].to_numpy().astype(bool)
    s = np.asarray(score, dtype=np.float64)
    order = np.lexsort((tie_key(meta["row_id"].to_numpy(), seed), -s, imp))
    imp_o, y_o, cold_o = imp[order], y[order], cold[order]
    starts = np.flatnonzero(np.r_[True, imp_o[1:] != imp_o[:-1]])
    group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(imp_o)]))
    rank = np.arange(len(imp_o)) - starts[group] + 1          # 1-based rank in its impression

    n_imp = len(starts)
    n = np.bincount(group, minlength=n_imp)
    pos = np.bincount(group, weights=y_o, minlength=n_imp)
    out = pd.DataFrame({"impression_id": imp_o[starts], "n_candidates": n, "n_clicks": pos})

    with np.errstate(invalid="ignore", divide="ignore"):
        out["mrr"] = np.bincount(group, weights=y_o / rank, minlength=n_imp) / pos
        disc = 1.0 / np.log2(rank + 1)
        ideal_cum = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(2, n.max() + 2)))]
        for k in ks:
            dcg = np.bincount(group, weights=y_o * disc * (rank <= k), minlength=n_imp)
            out[f"ndcg@{k}"] = dcg / ideal_cum[np.minimum(pos, k).astype(int)]

        # AUC with average ranks for tied scores (ascending ranks within impression)
        s_o = s[order]
        avg_rank = pd.Series(s_o).groupby(group).rank(method="average").to_numpy()
        neg = n - pos
        auc = (np.bincount(group, weights=avg_rank * y_o, minlength=n_imp) - pos * (pos + 1) / 2) / (pos * neg)
        out["auc"] = np.where((pos > 0) & (neg > 0), auc, np.nan)

        for name, mask in (("cold", cold_o), ("warm", ~cold_o)):
            clicked = y_o * mask
            n_clicked = np.bincount(group, weights=clicked, minlength=n_imp)
            out[f"n_{name}_clicks"] = n_clicked
            for k in ks:
                hit = np.bincount(group, weights=clicked * (rank <= k), minlength=n_imp)
                out[f"{name}_recall@{k}"] = np.where(n_clicked > 0, hit / n_clicked, np.nan)
                if name == "cold":
                    out[f"cold_hit@{k}"] = np.where(n_clicked > 0, (hit > 0).astype(float), np.nan)
    return out


def summarise(per_imp: pd.DataFrame) -> dict:
    """Mean of every metric over eligible impressions, plus support counts."""
    metric_cols = [c for c in per_imp.columns if "@" in c or c in ("auc", "mrr")]
    res = {c: float(per_imp[c].mean()) for c in metric_cols}
    res["n_impressions"] = int(len(per_imp))
    res["n_auc_impressions"] = int(per_imp["auc"].notna().sum())
    res["n_cold_click_impressions"] = int((per_imp["n_cold_clicks"] > 0).sum())
    res["n_cold_clicks"] = int(per_imp["n_cold_clicks"].sum())
    return res


def evaluate(meta: pd.DataFrame, score, **kwargs) -> dict:
    return summarise(per_impression_metrics(meta, score, **kwargs))


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 1000, seed: int = 0) -> dict:
    """Mean difference b - a over impressions eligible for both, with a 95% bootstrap interval."""
    d = (b - a).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1) if len(d) else np.array([np.nan])
    return {"mean_diff": float(d.mean()) if len(d) else np.nan, "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)), "n_impressions": int(len(d))}
