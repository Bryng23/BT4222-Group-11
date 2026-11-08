"""Balance, stress-slate, error-analysis, business and ablation helpers. Owner: Shi Ting.

Everything works on saved scores: no model is retrained here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluator as E
from . import feature_groups as FG


# ---------------------------------------------------------------------------
# Balance: slot-quota re-ranking and the trade-off rule
# ---------------------------------------------------------------------------
def slot_quota_rerank(meta: pd.DataFrame, score, cold, q: float, k: int = 10,
                      seed: int = E.TIE_SEED) -> np.ndarray:
    """Reserve round(q * k) of the top-k slots for the best-scoring cold candidates.

    The top k then holds those reserved cold candidates plus the best-scoring of the
    rest; inside the top k, and below it, the original order is kept. Slates of k or
    fewer candidates are unchanged. Returns new scores in meta row order (higher = better)
    that reproduce this order exactly, so they can go straight into the evaluator.
    The new scores have no ties, so AUC-type metrics (which count ties as 1/2) can
    move slightly even at q=0; read rank metrics (Recall, nDCG, slot share) from the sweep.
    """
    r = E.Ranked(meta, score, seed)
    cold_o = r.sorted(np.asarray(cold, dtype=bool))
    m = int(round(q * k))
    cold_cum = np.cumsum(cold_o)
    cold_rank = cold_cum - (cold_cum[r.starts] - cold_o[r.starts])[r.group]
    promoted = cold_o & (cold_rank <= m) & (r.n[r.group] > k)
    n_prom = r.sum(promoted)[r.group]
    rest = ~promoted
    rest_cum = np.cumsum(rest)
    rest_rank = rest_cum - (rest_cum[r.starts] - rest[r.starts])[r.group]
    in_top = promoted | (rest & (rest_rank <= k - n_prom))
    new_pos = r.rank + (~in_top) * (r.n.max() + 1)
    return r.unsort(-new_pos.astype(np.float64))


def quota_sweep(meta: pd.DataFrame, score, quotas=(0.0, 0.1, 0.2, 0.3), cold_threshold: int = 10,
                k: int = 10) -> tuple[pd.DataFrame, dict]:
    """Metrics after slot-quota re-ranking at each quota. Returns (summary table, per-impression tables)."""
    cold = E.cold_mask(meta, cold_threshold)
    rows, per_imps = {}, {}
    for q in quotas:
        s = score if q == 0 else slot_quota_rerank(meta, score, cold, q, k)
        pi = E.per_impression_metrics(meta, s, cold_threshold=cold_threshold)
        per_imps[q] = pi
        rows[q] = E.summarise(pi)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "quota"
    return table, per_imps


def pareto_front(table: pd.DataFrame, x: str = "warm_recall@10", y: str = "cold_recall@10") -> pd.Series:
    """True for rows no other row beats on both x and y."""
    v = table[[x, y]].to_numpy()
    dominated = [(np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1)).any() for i in range(len(v))]
    return pd.Series(~np.array(dominated), index=table.index, name="pareto")


def balance_select(table: pd.DataFrame, ref_warm: float, max_warm_drop: float,
                   cold: str = "cold_recall@10", warm: str = "warm_recall@10"):
    """Proposed balance rule: highest cold metric with warm metric >= ref_warm * (1 - max_warm_drop).

    Returns the selected index label, or None when no row satisfies the warm guard.
    This is a proposal for the team (conflict #5); selection.py is not changed.
    """
    ok = table[table[warm] >= ref_warm * (1 - max_warm_drop)]
    return None if ok.empty else ok.sort_values([cold, warm], ascending=False).index[0]


# ---------------------------------------------------------------------------
# Stress slates
# ---------------------------------------------------------------------------
def stress_slate_mask(meta: pd.DataFrame, cold, slate_size: int = 10, warm_share: float = 0.4,
                      seed: int = 0) -> tuple[np.ndarray, dict]:
    """Rebuild every impression as a slate with a fixed warm/cold mix.

    All clicks are kept; unclicked candidates of each type are sampled (seeded) to fill
    round(warm_share * slate_size) warm and the remainder cold slots. Impressions that
    cannot meet the mix (too many clicks of one type, too few negatives of a type) are
    dropped and counted. Valid only for pointwise scores: each candidate's score must not
    depend on which other candidates are in the slate.
    """
    cold = np.asarray(cold, dtype=bool)
    y = meta["label"].to_numpy().astype(bool)
    imp = meta["impression_id"].to_numpy()
    n_warm_t = int(round(warm_share * slate_size))
    target = np.where(cold, slate_size - n_warm_t, n_warm_t)
    d = pd.DataFrame({"imp": imp, "cold": cold, "y": y, "neg": ~y, "u": np.random.default_rng(seed).random(len(meta))})
    by_type = d.groupby(["imp", "cold"])
    clicks = by_type["y"].transform("sum").to_numpy()
    negs = by_type["neg"].transform("sum").to_numpy()
    need = target - clicks
    neg_order = d[~y].groupby(["imp", "cold"])["u"].rank(method="first")
    keep_neg = np.zeros(len(d), dtype=bool)
    keep_neg[np.flatnonzero(~y)] = neg_order.to_numpy() <= need[~y]
    # impression is valid if, for both types, clicks <= target and negatives >= target - clicks
    ok_row = (need >= 0) & (negs >= need)
    cnt = d.assign(ok=ok_row).groupby("imp").agg(ok=("ok", "all"), types=("cold", "nunique"), clicks=("y", "sum"))
    # both types must be present unless one type's target is zero
    need_types = 2 if 0 < n_warm_t < slate_size else 1
    valid = cnt.index[cnt["ok"] & (cnt["types"] >= need_types) & (cnt["clicks"] > 0)]
    mask = np.isin(imp, valid) & (y | keep_neg)
    info = {"slate_size": slate_size, "warm_share": warm_share, "seed": seed,
            "impressions_in": int(len(cnt)), "impressions_kept": int(len(valid)),
            "share_kept": float(len(valid) / max(len(cnt), 1)), "rows_kept": int(mask.sum())}
    return mask, info


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------
CASES = {
    "success": "clicked cold candidate ranked in the top 3",
    "failure": "clicked cold candidate ranked in the bottom quarter of its slate",
    "false_alarm": "unclicked cold candidate ranked in the top 3",
}


def error_cases(meta: pd.DataFrame, score, cold, attrs: pd.DataFrame | None = None,
                seed: int = E.TIE_SEED) -> pd.DataFrame:
    """Every cold candidate in one of the three error-analysis cases, with rank and attributes.

    attrs: optional table keyed by row_id (e.g. entity_count, article_age_hours, title,
    abstract) joined onto the cases. The comparison uses all cases, not only the sample.
    """
    r = E.Ranked(meta, score, seed)
    rank = r.unsort(r.rank)
    n = r.unsort(r.n[r.group])
    cold = np.asarray(cold, dtype=bool)
    y = meta["label"].to_numpy().astype(bool)
    case = np.full(len(meta), None, dtype=object)
    case[cold & y & (rank <= 3)] = "success"
    case[cold & y & (rank > 0.75 * n)] = "failure"
    case[cold & ~y & (rank <= 3)] = "false_alarm"
    keep = case != None  # noqa: E711  (elementwise on an object array)
    cols = [c for c in ("row_id", "impression_id", "user_id", "news_id", "category", "subcategory",
                        "prior_exposures", "history_length", "time") if c in meta.columns]
    out = meta.loc[keep, cols].copy()
    out["case"] = case[keep]
    out["rank"] = rank[keep]
    out["slate_size"] = n[keep]
    out["rank_pct"] = out["rank"] / out["slate_size"]
    if "time" in out.columns:
        out["hour"] = pd.to_datetime(out["time"]).dt.hour
    if attrs is not None:
        out = out.merge(attrs, on="row_id", how="left", suffixes=("", "_attr"))
    return out.reset_index(drop=True)


def compare_cases(cases: pd.DataFrame, numeric=None, categorical=("category",), top: int = 6) -> dict:
    """Pattern tables: numeric attributes by case (mean, median) and category mix by case."""
    numeric = numeric or [c for c in ("slate_size", "history_length", "prior_exposures", "entity_count",
                                      "article_age_hours", "abstract_present", "title_length", "hour")
                          if c in cases.columns]
    num = cases.groupby("case")[numeric].agg(["mean", "median"])
    num.insert(0, ("n", ""), cases.groupby("case").size())
    out = {"numeric": num}
    for c in categorical:
        if c in cases.columns:
            mix = pd.crosstab(cases[c], cases["case"], normalize="columns")
            out[c] = mix.loc[mix.max(axis=1).sort_values(ascending=False).index[:top]]
    return out


def sample_cases(cases: pd.DataFrame, n: int = 30, seed: int = 0) -> dict:
    """Seeded sample of n rows per case, for reading titles. False alarms are taken from rank 1 first."""
    out = {}
    for name, d in cases.groupby("case"):
        if name == "false_alarm":
            d = d.sort_values("rank", kind="stable").head(max(n * 5, n))
        out[name] = d.sample(min(n, len(d)), random_state=seed)
    return out


# ---------------------------------------------------------------------------
# Business numbers
# ---------------------------------------------------------------------------
def business_impact(per_imp_base: pd.DataFrame, per_imp_model: pd.DataFrame, k: int = 10,
                    n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Clicked cold / warm candidates placed in the top k, per 1,000 impressions, model minus base.

    Uses n_cold_hits@k and n_warm_hits@k from the evaluator; intervals are user-clustered.
    These are offline counts on logged slates, not predicted online clicks.
    """
    cols = [f"n_cold_hits@{k}", f"n_warm_hits@{k}", f"n_zero_hits@{k}"]
    cols = [c for c in cols if c in per_imp_base.columns]
    a = per_imp_base[["impression_id", "user_id"] + cols].copy() if "user_id" in per_imp_base else per_imp_base[["impression_id"] + cols].copy()
    b = a.copy()
    b[cols] = per_imp_model[cols].to_numpy()
    res = E.cluster_bootstrap(a.astype({c: float for c in cols}), b.astype({c: float for c in cols}), cols,
                              n_boot=n_boot, seed=seed)
    res[["estimate", "ci_low", "ci_high", "se"]] *= 1000
    res.insert(0, "base_per_1000", [1000 * a[c].mean() for c in cols])
    res.insert(1, "model_per_1000", [1000 * b[c].mean() for c in cols])
    return res.rename(columns={"estimate": "delta_per_1000"})


# ---------------------------------------------------------------------------
# Ablation helpers (group-agnostic: read from feature_groups.ABLATIONS)
# ---------------------------------------------------------------------------
def tier_matrix(variants=None) -> pd.DataFrame:
    """Which groups each variant uses, from FG.ABLATIONS. New variants or groups need no code change."""
    variants = variants or list(FG.ABLATIONS)
    groups = sorted({g for v in variants for g in FG.ABLATIONS.get(v, [])})
    return pd.DataFrame([[g in FG.ABLATIONS.get(v, []) for g in groups] for v in variants], index=variants, columns=groups)


def describe_variant(variant: str, base: str = "full", reference: str = "u_only") -> dict:
    """Kind of comparison a variant supports: leave-one-out against base, add-one-in against reference."""
    v, b = set(FG.ABLATIONS.get(variant, [])), set(FG.ABLATIONS[base])
    ref = set(FG.ABLATIONS.get(reference, ["U"]))
    if variant == base:
        return {"kind": "base", "groups": ""}
    if v < b:
        return {"kind": "leave-out", "groups": "+".join(sorted(b - v))}
    if v > b:
        return {"kind": "add-on", "groups": "+".join(sorted(v - b))}
    if v > ref:
        return {"kind": "add-in", "groups": "+".join(sorted(v - ref))}
    return {"kind": "other", "groups": "+".join(sorted(v))}


def ablation_deltas(per_imps: dict, base: str, metrics, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Every variant minus base on the same impressions, user-clustered intervals. Long format."""
    rows = []
    for variant, pi in per_imps.items():
        if variant == base:
            continue
        d = E.compare(per_imps[base], pi, metrics, n_boot=n_boot, seed=seed)
        d = d.reset_index().assign(variant=variant, against=base)
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def decision(val_ci_low: float, val_ci_high: float, test_delta: float | None) -> str:
    """Pre-registered rule for a leave-one-out row (delta = variant - full on cold Recall@10).

    "tier helps cold": removing the tier lowers the metric, validation CI entirely below 0,
    and the test delta has the same sign. "tier hurts cold": the mirror image.
    Anything else: "no detectable effect". test_delta None means test not yet read.
    """
    if val_ci_high < 0 and (test_delta is None or test_delta < 0):
        return "tier helps cold" + (" (val only)" if test_delta is None else "")
    if val_ci_low > 0 and (test_delta is None or test_delta > 0):
        return "tier hurts cold" + (" (val only)" if test_delta is None else "")
    return "no detectable effect"


def seed_spread(summary: pd.DataFrame, metrics, by: str = "variant") -> pd.DataFrame:
    """Mean, SD, min and max across training seeds; summary has one row per (variant, seed)."""
    return summary.groupby(by)[list(metrics)].agg(["mean", "std", "min", "max", "count"])


# ---------------------------------------------------------------------------
# Plain-language readings (used by the research-question summary and the report)
# ---------------------------------------------------------------------------
def direction(lo: float, hi: float, up: str = "higher", down: str = "lower",
              none: str = "no detectable change") -> str:
    """Word for an interval: up if it is entirely above 0, down if entirely below, else none."""
    if lo > 0:
        return up
    if hi < 0:
        return down
    return none


def tradeoff_verdict(cold_lo: float, cold_hi: float, warm_lo: float, warm_hi: float) -> str:
    """One sentence on both sides of the trade-off, from two intervals (model minus comparison)."""
    c = direction(cold_lo, cold_hi, "better for cold articles", "worse for cold articles",
                  "no detectable difference for cold articles")
    w = direction(warm_lo, warm_hi, "better for warm articles", "worse for warm articles",
                  "no detectable difference for warm articles")
    return f"{c}; {w}"


def tier_contributions(abl_delta: pd.DataFrame, split: str, base: str = "full",
                       metrics=("cold_recall@10", "warm_recall@10", "zero_recall@10", "cold_auc_within", "ndcg@10")
                       ) -> pd.DataFrame:
    """Contribution of each removed group = full minus the variant without it (positive = the group helps).

    Built from the leave-one-out rows of ablation_deltas (variant - full), sign flipped so the
    table reads "what the group adds". One row per leave-out variant.
    """
    d = abl_delta[(abl_delta["split"] == split) & (abl_delta["against"] == base) & abl_delta["metric"].isin(metrics)]
    rows = []
    for v, g in d.groupby("variant", sort=False):
        info = describe_variant(v, base)
        if info["kind"] != "leave-out":
            continue
        row = {"variant": v, "group removed": info["groups"]}
        for _, r in g.iterrows():
            m = r["metric"]
            row[f"{m} contribution"] = -r["delta"]
            row[f"{m} ci_low"], row[f"{m} ci_high"] = -r["ci_high"], -r["ci_low"]
            row[f"{m} reading"] = direction(-r["ci_high"], -r["ci_low"], "helps", "hurts", "no detectable effect")
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Rich-get-richer: how concentrated is exposure?
# ---------------------------------------------------------------------------
def popularity_groups(meta: pd.DataFrame, top: float = 0.01, bottom: float = 0.8) -> np.ndarray:
    """Per row: 'top', 'middle' or 'bottom' by the article's logged appearances in this split.

    Articles are ordered by how many candidate rows they have (most first, ties by news_id);
    the first `top` share of articles is 'top', the last `bottom` share is 'bottom'.
    """
    counts = meta["news_id"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable")
    n = len(counts)
    n_top, n_bottom = max(1, int(np.ceil(top * n))), int(np.floor(bottom * n))
    lab = np.full(n, "middle", dtype=object)
    lab[:n_top] = "top"
    lab[n - n_bottom:] = "bottom"
    return meta["news_id"].map(pd.Series(lab, index=counts.index)).to_numpy()


def exposure_concentration(meta: pd.DataFrame, scores: dict, k: int = 10, top: float = 0.01,
                           bottom: float = 0.8, seed: int = E.TIE_SEED) -> pd.DataFrame:
    """Share of exposure going to the most- and least-shown articles: logged vs each model's top k.

    'logged' = share of all candidate rows in the split (what MSN actually showed).
    Each model column = share of all top-k slots across impressions.
    """
    g = popularity_groups(meta, top, bottom)
    out = {"logged": pd.Series(g).value_counts(normalize=True)}
    for name, s in scores.items():
        r = E.Ranked(meta, s, seed)
        out[name] = pd.Series(r.sorted(g)[r.rank <= k]).value_counts(normalize=True)
    res = pd.DataFrame(out).reindex(["top", "middle", "bottom"]).fillna(0.0)
    res.index = [f"top {top:.0%} of articles", "middle", f"bottom {bottom:.0%} of articles"]
    return res


# ---------------------------------------------------------------------------
# Error analysis by evidence tier
# ---------------------------------------------------------------------------
TIER_ATTRS = {
    "T1 own behaviour": ["prior_exposures", "prior_ctr", "recent_exposures_6h", "article_age_hours"],
    "T2 borrowed behaviour": ["category_recent_ctr_6h", "subcategory_recent_ctr_6h", "entity_recent_ctr_mean_6h",
                              "semantic_neighbor_ctr"],
    "T3 content": ["entity_count", "entity_embedding_coverage", "abstract_present", "title_length"],
    "U user matching": ["history_length", "category_affinity", "subcategory_affinity", "text_similarity",
                        "entity_similarity"],
    "Slate context": ["slate_size", "hour"],
}


ATTR_LABELS = {
    "prior_exposures": "prior exposures", "prior_ctr": "prior CTR", "recent_exposures_6h": "exposures in the last 6 h",
    "article_age_hours": "article age (hours)", "category_recent_ctr_6h": "recent CTR of its category",
    "subcategory_recent_ctr_6h": "recent CTR of its subcategory", "entity_recent_ctr_mean_6h": "recent CTR of its entities",
    "semantic_neighbor_ctr": "CTR of similar articles", "entity_count": "number of entities",
    "entity_embedding_coverage": "entity embedding coverage", "abstract_present": "share with an abstract",
    "title_length": "title length (words)", "history_length": "reader history length",
    "category_affinity": "reader's affinity for the category", "subcategory_affinity": "reader's affinity for the subcategory",
    "text_similarity": "text similarity to the reader's history", "entity_similarity": "entity similarity to the reader's history",
    "slate_size": "slate length", "hour": "hour of day",
}


def tier_patterns(cases: pd.DataFrame, a: str = "success", b: str = "failure", tier_attrs=TIER_ATTRS,
                  notable: float = 0.2) -> pd.DataFrame:
    """Which evidence differs between successes and failures, grouped by tier.

    std_diff = (mean in b - mean in a) / pooled SD; |std_diff| >= notable (0.2, a "small" effect)
    is flagged. Means for every case type are shown so false alarms can be read too.
    """
    rows = []
    case_names = list(dict.fromkeys(cases["case"]))
    for tier, cols in tier_attrs.items():
        for c in cols:
            if c not in cases.columns:
                continue
            v = cases[c].astype(float)
            x, y = v[cases["case"] == a].dropna(), v[cases["case"] == b].dropna()
            sd = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2) if len(x) > 1 and len(y) > 1 else np.nan
            d = (y.mean() - x.mean()) / sd if sd and sd > 0 else np.nan
            row = {"tier": tier, "attribute": c, **{f"mean {n}": v[cases["case"] == n].mean() for n in case_names},
                   "std_diff": d, "notable": bool(np.isfinite(d) and abs(d) >= notable)}
            row["reading"] = ("" if not np.isfinite(d) else
                              f"{b} cases have {'higher' if d > 0 else 'lower'} {ATTR_LABELS.get(c, c)} than {a} cases "
                              f"({y.mean():.3g} vs {x.mean():.3g})")
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.assign(_abs=out["std_diff"].abs()).sort_values(["notable", "_abs"], ascending=False).drop(columns="_abs")
    return out.reset_index(drop=True)
