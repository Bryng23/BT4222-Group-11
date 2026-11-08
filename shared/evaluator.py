"""Team evaluator (replaces shared/eval_interim.py). Owner: Shi Ting.

Same public API as eval_interim (tie_key, per_impression_metrics, summarise,
evaluate, paired_bootstrap), so a notebook switches with one line:

    from shared import evaluator as E

Every definition is written out in docs/evaluation_spec.md (metric dictionary)
and checked against a brute-force implementation in tests/test_evaluator.py.

Conventions
-----------
* Ranking is per impression by score, descending. Ties in the ranking are broken
  by a seeded hash of row_id (tie_key), which ignores labels and is identical for
  every model. AUC-type metrics instead count tied scores as 1/2, as in MIND.
* "Cold" at threshold t means prior_exposures < t at the impression time. The
  flag column is_cold_{t} is used when it exists (contract v0.1); otherwise the
  flag is computed from prior_exposures. audit.check_cold_flags verifies the two agree.
* Every metric is a per-impression value, NaN where the impression is not
  eligible, then averaged over eligible impressions (macro average).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_SEED = 4222
KS = (5, 10)
GAMMAS = (0.5, 1.0)
SLOT_K = 10

# Impression-level attributes carried into the per-impression table when present,
# so segments and the cluster bootstrap need no second join.
CARRY_COLUMNS = ("user_id", "time", "history_length", "split")

# Columns of the per-impression table that are not metrics.
ID_COLUMNS = {"impression_id", *CARRY_COLUMNS}


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
def tie_key(row_id, seed: int = TIE_SEED) -> np.ndarray:
    """Deterministic pseudo-random key per row (splitmix64 of row_id + seed). Identical to eval_interim."""
    with np.errstate(over="ignore"):
        z = np.asarray(row_id, dtype=np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


class Ranked:
    """Rows sorted by (impression, score desc, tie key), with 1-based rank per impression.

    Shared by the evaluator, segments and diagnostics so the ordering is computed once.
    All arrays ending in _o are in sorted order; `order` maps sorted position -> meta row.
    """

    def __init__(self, meta: pd.DataFrame, score, seed: int = TIE_SEED):
        s = np.asarray(score, dtype=np.float64)
        if len(s) != len(meta):
            raise ValueError(f"score has {len(s)} rows, meta has {len(meta)}")
        if not np.isfinite(s).all():
            raise ValueError("non-finite scores; run audit.check_predictions first")
        imp = meta["impression_id"].to_numpy()
        self.order = np.lexsort((tie_key(meta["row_id"].to_numpy(), seed), -s, imp))
        self.imp_o = imp[self.order]
        self.s_o = s[self.order]
        self.y_o = meta["label"].to_numpy().astype(np.float64)[self.order]
        self.starts = np.flatnonzero(np.r_[True, self.imp_o[1:] != self.imp_o[:-1]])
        self.n_imp = len(self.starts)
        self.group = np.repeat(np.arange(self.n_imp), np.diff(np.r_[self.starts, len(self.imp_o)]))
        self.rank = np.arange(len(self.imp_o)) - self.starts[self.group] + 1
        self.n = np.bincount(self.group, minlength=self.n_imp)
        self.pos = self.sum(self.y_o)

    def sum(self, w) -> np.ndarray:
        """Per-impression sum of a sorted-order weight array."""
        return np.bincount(self.group, weights=np.asarray(w, dtype=np.float64), minlength=self.n_imp)

    def sorted(self, values) -> np.ndarray:
        """Reorder a meta-aligned array into sorted order."""
        return np.asarray(values)[self.order]

    def unsort(self, values_o) -> np.ndarray:
        """Inverse of sorted(): back to meta row order."""
        out = np.empty_like(values_o)
        out[self.order] = values_o
        return out

    def first(self, values) -> np.ndarray:
        """Impression-level value taken from each impression's first sorted row."""
        return self.sorted(values)[self.starts]

    def neg_below(self, neg_mask_o) -> tuple[np.ndarray, np.ndarray]:
        """For every row: negatives (within neg_mask) scored strictly lower + 1/2 the tied ones.

        Returns (per-row count, per-impression number of negatives in the mask).
        Summed over positives and divided by n_pos * n_neg this is the Mann-Whitney AUC,
        which equals the average-rank AUC used by eval_interim.
        """
        negw = ((1 - self.y_o) * neg_mask_o).astype(np.float64)
        n_neg = self.sum(negw)
        new_block = np.r_[True, (self.imp_o[1:] != self.imp_o[:-1]) | (self.s_o[1:] != self.s_o[:-1])]
        block = np.cumsum(new_block) - 1
        block_starts = np.flatnonzero(new_block)
        block_end = np.r_[block_starts[1:], len(self.s_o)] - 1
        csum = np.cumsum(negw)
        before_imp = csum[self.starts] - negw[self.starts]          # global cumsum before each impression
        through_block = csum[block_end][block] - before_imp[self.group]
        in_block = np.bincount(block, weights=negw)[block]
        return n_neg[self.group] - through_block + 0.5 * in_block, n_neg


def cold_mask(meta: pd.DataFrame, threshold: int = 10) -> np.ndarray:
    """Candidate is cold if prior_exposures < threshold (flag column preferred when present)."""
    col = f"is_cold_{threshold}"
    if col in meta.columns:
        return meta[col].to_numpy().astype(bool)
    if "prior_exposures" not in meta.columns:
        raise KeyError(f"meta has neither {col} nor prior_exposures")
    return meta["prior_exposures"].to_numpy() < threshold


def zero_mask(meta: pd.DataFrame) -> np.ndarray:
    if "is_zero_exposure" in meta.columns:
        return meta["is_zero_exposure"].to_numpy().astype(bool)
    return meta["prior_exposures"].to_numpy() == 0


def _threshold_from_col(cold_col: str) -> int:
    return int(cold_col.rsplit("_", 1)[-1])


# ---------------------------------------------------------------------------
# Per-impression metrics
# ---------------------------------------------------------------------------
def per_impression_metrics(meta: pd.DataFrame, score, cold_col: str = "is_cold_10", ks=KS,
                           seed: int = TIE_SEED, *, cold_threshold: int | None = None,
                           gammas=GAMMAS, slot_k: int = SLOT_K, extended: bool = True,
                           ranked: Ranked | None = None) -> pd.DataFrame:
    """One row per impression with every metric (NaN where the impression is not eligible).

    meta needs row_id, impression_id, label and either the cold flag column or
    prior_exposures. cold_threshold, when given, overrides cold_col.
    extended=False returns only the eval_interim columns (faster for tuning loops).
    """
    t = cold_threshold if cold_threshold is not None else _threshold_from_col(cold_col)
    r = ranked if ranked is not None else Ranked(meta, score, seed)
    y, rank, n, pos = r.y_o, r.rank, r.n, r.pos
    cold_o = r.sorted(cold_mask(meta, t))

    out = pd.DataFrame({"impression_id": r.imp_o[r.starts], "n_candidates": n, "n_clicks": pos})
    for c in CARRY_COLUMNS:
        if c in meta.columns:
            out[c] = r.first(meta[c].to_numpy())

    with np.errstate(invalid="ignore", divide="ignore"):
        # --- MIND-standard ---------------------------------------------------
        neg_below_all, n_neg = r.neg_below(np.ones_like(y))
        out["auc"] = np.where((pos > 0) & (n_neg > 0), r.sum(y * neg_below_all) / (pos * n_neg), np.nan)
        out["mrr"] = np.where(pos > 0, r.sum(y / rank) / pos, np.nan)
        disc = 1.0 / np.log2(rank + 1)
        ideal_cum = np.r_[0.0, np.cumsum(1.0 / np.log2(np.arange(2, n.max() + 2)))]
        for k in ks:
            out[f"ndcg@{k}"] = r.sum(y * disc * (rank <= k)) / ideal_cum[np.minimum(pos, k).astype(int)]

        # --- cold / warm / zero-exposure recall in the full slate -------------
        masks = {"cold": cold_o, "warm": ~cold_o}
        if extended:
            masks["zero"] = r.sorted(zero_mask(meta))
        for name, mask in masks.items():
            clicked = y * mask
            n_clicked = r.sum(clicked)
            out[f"n_{name}_clicks"] = n_clicked
            for k in ks:
                hits = r.sum(clicked * (rank <= k))
                out[f"{name}_recall@{k}"] = np.where(n_clicked > 0, hits / n_clicked, np.nan)
                if name == "cold":
                    out[f"cold_hit@{k}"] = np.where(n_clicked > 0, (hits > 0).astype(float), np.nan)
                if extended:
                    out[f"n_{name}_hits@{k}"] = hits
        if not extended:
            return out

        # --- cold AUC, two variants -------------------------------------------
        n_cold_pos = out["n_cold_clicks"].to_numpy()
        nb_cold, n_cold_neg = r.neg_below(cold_o.astype(float))
        out["n_cold_negatives"] = n_cold_neg
        within_ok = (n_cold_pos > 0) & (n_cold_neg > 0)
        out["cold_auc_within"] = np.where(within_ok, r.sum(y * cold_o * nb_cold) / (n_cold_pos * n_cold_neg), np.nan)
        out["cold_auc_vs_all"] = np.where((n_cold_pos > 0) & (n_neg > 0),
                                          r.sum(y * cold_o * neg_below_all) / (n_cold_pos * n_neg), np.nan)

        # --- cold-only nDCG: warm candidates dropped, cold keep their relative order
        cold_cum = np.cumsum(cold_o)
        cold_rank = cold_cum - (cold_cum[r.starts] - cold_o[r.starts])[r.group]   # 1-based among cold
        cold_disc = np.where(cold_o, 1.0 / np.log2(np.maximum(cold_rank, 1) + 1), 0.0)
        for k in ks:
            dcg = r.sum(y * cold_disc * (cold_rank <= k))
            ndcg = dcg / ideal_cum[np.minimum(n_cold_pos, k).astype(int)]
            out[f"cold_ndcg@{k}"] = np.where(within_ok, ndcg, np.nan)

        # --- popularity-debiased nDCG -----------------------------------------
        prior = r.sorted(meta["prior_exposures"].to_numpy()).astype(np.float64)
        pos_idx = np.flatnonzero(y > 0)
        for g in gammas:
            w = y / np.power(prior + 1.0, g)
            # ideal: positives in descending weight; position among positives of the impression
            idx = pos_idx[np.lexsort((-w[pos_idx], r.group[pos_idx]))]
            gi = r.group[idx]
            first_in_imp = np.r_[True, gi[1:] != gi[:-1]]
            j = np.arange(len(idx)) - np.flatnonzero(first_in_imp)[np.cumsum(first_in_imp) - 1] + 1
            for k in ks:
                idcg = np.bincount(gi, weights=w[idx] * (j <= k) / np.log2(j + 1), minlength=r.n_imp)
                dcg = r.sum(w * disc * (rank <= k))
                out[f"popdebiased_ndcg@{k}_g{g:g}"] = np.where(pos > 0, dcg / idcg, np.nan)

        # --- exposure of cold items: slot share vs candidate share ------------
        top = rank <= slot_k
        out[f"cold_slot_share@{slot_k}"] = r.sum(cold_o * top) / np.minimum(n, slot_k)
        out["cold_candidate_share"] = r.sum(cold_o) / n

        # --- novelty of the top-k: mean -log2 popularity -----------------------
        if "news_id" in meta.columns:
            nid = meta["news_id"].to_numpy()
            n_imps_with = pd.Series(meta["impression_id"].to_numpy()).groupby(nid).nunique()
            p = pd.Series(nid).map(n_imps_with).to_numpy(np.float64) / r.n_imp
            out[f"novelty@{slot_k}"] = r.sum(-np.log2(r.sorted(p)) * top) / np.minimum(n, slot_k)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def metric_columns(per_imp: pd.DataFrame) -> list[str]:
    """Columns of a per-impression table that are metrics (not ids, counts or attributes)."""
    return [c for c in per_imp.columns
            if c not in ID_COLUMNS and not c.startswith("n_") and pd.api.types.is_float_dtype(per_imp[c])]


def summarise(per_imp: pd.DataFrame) -> dict:
    """Mean of every metric over its eligible impressions, lifts over random, and support counts.

    Lift over random for Recall@k uses the exact expectation for a random ranker,
    min(k, n) / n per impression, averaged over the same eligible impressions.
    """
    res = {c: float(per_imp[c].mean()) for c in metric_columns(per_imp)}
    n = per_imp["n_candidates"].to_numpy(np.float64)
    for c in [c for c in per_imp.columns if "_recall@" in c]:
        k = int(c.rsplit("@", 1)[1])
        ok = per_imp[c].notna().to_numpy()
        res[f"{c}_random"] = float(np.mean(np.minimum(k, n[ok]) / n[ok])) if ok.any() else np.nan
        res[f"{c}_lift"] = res[c] - res[f"{c}_random"]
    if "auc" in res:
        res["auc_lift"] = res["auc"] - 0.5
    if "cold_slot_share@10" in res and res.get("cold_candidate_share"):
        res["cold_exposure_ratio@10"] = res["cold_slot_share@10"] / res["cold_candidate_share"]

    res["n_impressions"] = int(len(per_imp))
    res["n_clicks"] = int(per_imp["n_clicks"].sum())
    res["n_auc_impressions"] = int(per_imp["auc"].notna().sum())
    for name in ("cold", "warm", "zero"):
        if f"n_{name}_clicks" in per_imp.columns:
            res[f"n_{name}_click_impressions"] = int((per_imp[f"n_{name}_clicks"] > 0).sum())
            res[f"n_{name}_clicks"] = int(per_imp[f"n_{name}_clicks"].sum())
    if "cold_auc_within" in per_imp.columns:
        res["n_within_cold_impressions"] = int(per_imp["cold_auc_within"].notna().sum())
    if "user_id" in per_imp.columns:
        res["n_users"] = int(per_imp["user_id"].nunique())
    return res


def catalogue_coverage(meta: pd.DataFrame, score, k: int = SLOT_K, seed: int = TIE_SEED,
                       ranked: Ranked | None = None) -> float:
    """Distinct articles shown in any top-k / distinct articles among the split's candidates."""
    r = ranked if ranked is not None else Ranked(meta, score, seed)
    nid = r.sorted(meta["news_id"].to_numpy())
    return float(len(np.unique(nid[r.rank <= k])) / len(np.unique(nid)))


def evaluate(meta: pd.DataFrame, score, **kwargs) -> dict:
    """summarise(per_impression_metrics(...)) plus catalogue coverage when news_id is present."""
    seed = kwargs.get("seed", TIE_SEED)
    r = Ranked(meta, score, seed)
    res = summarise(per_impression_metrics(meta, score, ranked=r, **kwargs))
    if "news_id" in meta.columns and kwargs.get("extended", True):
        res[f"coverage@{SLOT_K}"] = catalogue_coverage(meta, score, ranked=r)
    return res


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def _cluster_codes(per_imp: pd.DataFrame, cluster: str | None) -> np.ndarray:
    if cluster is None or cluster not in per_imp.columns:
        return np.arange(len(per_imp))
    return pd.factorize(per_imp[cluster].to_numpy())[0]


def cluster_bootstrap(per_imp_a: pd.DataFrame, per_imp_b: pd.DataFrame | None = None, metrics=None, *,
                      cluster: str | None = "user_id", n_boot: int = 1000, seed: int = 0,
                      level: float = 0.95, chunk: int = 100) -> pd.DataFrame:
    """Bootstrap intervals that resample whole users (clusters), not impressions.

    One model (per_imp_b None): interval for the metric's mean.
    Two models: interval for mean(b - a) over impressions eligible in both (paired).
    The statistic is a ratio estimator, sum of values / number of eligible impressions,
    recomputed on each resample of clusters; all metrics share the same resamples.
    cluster=None gives the ordinary impression bootstrap.

    Returns one row per metric: estimate, ci_low, ci_high, se, n_impressions, n_clusters.
    """
    if per_imp_b is not None and not np.array_equal(per_imp_a["impression_id"].to_numpy(),
                                                      per_imp_b["impression_id"].to_numpy()):
        raise ValueError("per-impression tables are not aligned; evaluate both on the same meta")
    metrics = list(metrics or metric_columns(per_imp_a))
    codes = _cluster_codes(per_imp_a, cluster)
    n_cl = codes.max() + 1 if len(codes) else 0
    sums, counts, est, n_eligible, n_clusters = [], [], [], [], []
    for m in metrics:
        v = per_imp_a[m].to_numpy(np.float64)
        if per_imp_b is not None:
            v = per_imp_b[m].to_numpy(np.float64) - v
        ok = ~np.isnan(v)
        sums.append(np.bincount(codes[ok], weights=v[ok], minlength=n_cl))
        counts.append(np.bincount(codes[ok], minlength=n_cl).astype(np.float64))
        est.append(v[ok].mean() if ok.any() else np.nan)
        n_eligible.append(int(ok.sum()))
        n_clusters.append(int(len(np.unique(codes[ok]))))
    S, N = np.vstack(sums).T, np.vstack(counts).T                       # clusters x metrics
    rng = np.random.default_rng(seed)
    boots = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        W = np.vstack([np.bincount(rng.integers(0, n_cl, n_cl), minlength=n_cl) for _ in range(b)]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append((W @ S) / (W @ N))
    boots = np.vstack(boots)
    a = (1 - level) / 2
    lo, hi = np.nanpercentile(boots, [100 * a, 100 * (1 - a)], axis=0)
    return pd.DataFrame({"metric": metrics, "estimate": est, "ci_low": lo, "ci_high": hi,
                         "se": np.nanstd(boots, axis=0, ddof=1), "n_impressions": n_eligible,
                         "n_clusters": n_clusters}).set_index("metric")


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 1000, seed: int = 0,
                     clusters: pd.Series | None = None) -> dict:
    """Mean difference b - a over impressions eligible for both, with a 95% bootstrap interval.

    Same signature and output as eval_interim. Pass clusters (e.g. per_imp["user_id"])
    to resample users instead of impressions; the spec requires that for reported CIs.
    """
    frame = pd.DataFrame({"impression_id": np.arange(len(a)), "x": a.to_numpy(np.float64)})
    other = frame.assign(x=b.to_numpy(np.float64))
    if clusters is not None:
        frame["user_id"] = other["user_id"] = np.asarray(clusters)
    res = cluster_bootstrap(frame, other, ["x"], cluster="user_id" if clusters is not None else None,
                            n_boot=n_boot, seed=seed).loc["x"]
    return {"mean_diff": float(res["estimate"]), "ci_low": float(res["ci_low"]),
            "ci_high": float(res["ci_high"]), "n_impressions": int(res["n_impressions"])}


def compare(per_imp_base: pd.DataFrame, per_imp_other: pd.DataFrame, metrics, **kwargs) -> pd.DataFrame:
    """Paired, user-clustered deltas (other - base) for several metrics, plus both levels."""
    d = cluster_bootstrap(per_imp_base, per_imp_other, metrics, **kwargs)
    d.insert(0, "base", [per_imp_base[m].mean() for m in metrics])
    d.insert(1, "other", [per_imp_other[m].mean() for m in metrics])
    return d.rename(columns={"estimate": "delta"})
