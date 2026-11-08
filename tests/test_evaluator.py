"""Proof the evaluator is right: every vectorised metric equals a brute-force loop.

Run: python -m pytest tests/ -q
The synthetic slates include tied scores, impressions with no click, all-cold and
all-warm slates, and rows in shuffled order.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import audit as A  # noqa: E402
from shared import diagnostics as D  # noqa: E402
from shared import eval_interim as EI  # noqa: E402
from shared import evaluator as E  # noqa: E402
from shared import segments as SG  # noqa: E402


def make_meta(n_imp=80, seed=0, max_slate=25):
    rng = np.random.default_rng(seed)
    rows = []
    base_time = pd.Timestamp("2019-11-13")
    for i in range(n_imp):
        n = int(rng.integers(2, max_slate + 1))
        y = (rng.random(n) < 0.18).astype(int)
        if i % 9 != 0 and y.sum() == 0:   # most impressions have a click; some do not
            y[rng.integers(n)] = 1
        pe = rng.choice([0, 0, 1, 3, 4, 6, 9, 12, 19, 25, 60, 300], size=n)
        if i % 13 == 0:
            pe[:] = rng.integers(0, 5, n)      # all-cold slate
        if i % 17 == 0:
            pe[:] = rng.integers(30, 99, n)    # all-warm slate
        for j in range(n):
            rows.append({"impression_id": 1000 + i, "user_id": f"U{i % 23}", "news_id": f"N{rng.integers(0, 60)}",
                         "label": y[j], "prior_exposures": int(pe[j]), "history_length": int(i % 40),
                         "time": base_time + pd.Timedelta(minutes=17 * i), "category": f"c{rng.integers(0, 4)}"})
    m = pd.DataFrame(rows)
    m["row_id"] = rng.permutation(len(m)).astype(np.int64) + 5
    for t in (5, 10, 20):
        m[f"is_cold_{t}"] = m["prior_exposures"] < t
    m["is_zero_exposure"] = m["prior_exposures"] == 0
    m["slate_size"] = m.groupby("impression_id")["row_id"].transform("size")
    m = m.sample(frac=1, random_state=seed).reset_index(drop=True)          # rows not grouped by impression
    score = np.round(rng.normal(size=len(m)), 1)                            # many ties
    return m, score


# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------
def pair_auc(pos_scores, neg_scores):
    if not pos_scores or not neg_scores:
        return np.nan
    tot = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos_scores for q in neg_scores)
    return tot / (len(pos_scores) * len(neg_scores))


def dcg(gains_in_rank_order, k):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains_in_rank_order[:k]))


def brute(meta, score, t=10, ks=(5, 10), gammas=(0.5, 1.0)):
    keys = E.tie_key(meta["row_id"].to_numpy())
    d = meta.assign(s=score, key=keys)
    n_imps = d["impression_id"].nunique()
    pop = d.groupby("news_id")["impression_id"].nunique() / n_imps
    out = []
    for imp, g in d.groupby("impression_id"):
        recs = sorted(g.to_dict("records"), key=lambda r: (-r["s"], r["key"]))
        y = [r["label"] for r in recs]
        cold = [r["prior_exposures"] < t for r in recs]
        s = [r["s"] for r in recs]
        n, P = len(recs), sum(y)
        row = {"impression_id": imp}
        row["auc"] = pair_auc([s[i] for i in range(n) if y[i]], [s[i] for i in range(n) if not y[i]])
        row["mrr"] = np.mean([1 / (i + 1) for i in range(n) if y[i]]) if P else np.nan
        for k in ks:
            row[f"ndcg@{k}"] = dcg(y, k) / dcg(sorted(y, reverse=True), k) if P else np.nan
        for name, mask in (("cold", cold), ("warm", [not c for c in cold]),
                           ("zero", [r["prior_exposures"] == 0 for r in recs])):
            clicked = [i for i in range(n) if y[i] and mask[i]]
            for k in ks:
                hit = sum(1 for i in clicked if i < k)
                row[f"{name}_recall@{k}"] = hit / len(clicked) if clicked else np.nan
                if name == "cold":
                    row[f"cold_hit@{k}"] = float(hit > 0) if clicked else np.nan
        cp = [s[i] for i in range(n) if y[i] and cold[i]]
        cn = [s[i] for i in range(n) if not y[i] and cold[i]]
        an = [s[i] for i in range(n) if not y[i]]
        row["cold_auc_within"] = pair_auc(cp, cn)
        row["cold_auc_vs_all"] = pair_auc(cp, an)
        cold_y = [y[i] for i in range(n) if cold[i]]
        for k in ks:
            row[f"cold_ndcg@{k}"] = dcg(cold_y, k) / dcg(sorted(cold_y, reverse=True), k) if cp and cn else np.nan
        for gm in gammas:
            w = [y[i] / (recs[i]["prior_exposures"] + 1) ** gm for i in range(n)]
            for k in ks:
                row[f"popdebiased_ndcg@{k}_g{gm:g}"] = dcg(w, k) / dcg(sorted(w, reverse=True), k) if P else np.nan
        row["cold_slot_share@10"] = sum(cold[:10]) / min(n, 10)
        row["cold_candidate_share"] = sum(cold) / n
        row["novelty@10"] = np.mean([-math.log2(pop[r["news_id"]]) for r in recs[:10]])
        out.append(row)
    return pd.DataFrame(out).set_index("impression_id")


@pytest.fixture(scope="module", params=[0, 1, 2])
def case(request):
    meta, score = make_meta(seed=request.param)
    return meta, score, E.per_impression_metrics(meta, score).set_index("impression_id")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_every_metric_matches_brute_force(case):
    meta, score, fast = case
    ref = brute(meta, score)
    for col in ref.columns:
        pd.testing.assert_series_equal(fast[col].astype(float), ref[col].astype(float), check_names=False,
                                       atol=1e-12, obj=col)


def test_same_as_interim_on_shared_columns(case):
    meta, score, fast = case
    old = EI.per_impression_metrics(meta, score).set_index("impression_id")
    for col in old.columns:
        np.testing.assert_allclose(fast[col].to_numpy(float), old[col].to_numpy(float), atol=1e-12,
                                   equal_nan=True, err_msg=col)
    # summarise keeps eval_interim's keys and values
    s_old, s_new = EI.summarise(old.reset_index()), E.summarise(fast.reset_index())
    for k, v in s_old.items():
        assert s_new[k] == pytest.approx(v, nan_ok=True), k


def test_threshold_parameter_matches_flag_column():
    meta, score = make_meta(seed=3)
    for t in (5, 10, 20):
        a = E.per_impression_metrics(meta, score, cold_threshold=t)
        b = E.per_impression_metrics(meta.drop(columns=[f"is_cold_{t}"]), score, cold_threshold=t)
        c = E.per_impression_metrics(meta, score, cold_col=f"is_cold_{t}")
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(a, c)
    # a larger threshold can only add cold candidates
    s5, s20 = E.evaluate(meta, score, cold_threshold=5), E.evaluate(meta, score, cold_threshold=20)
    assert s20["n_cold_clicks"] >= s5["n_cold_clicks"]


def test_row_order_and_model_invariance():
    meta, score = make_meta(seed=4)
    a = E.per_impression_metrics(meta, score)
    perm = np.random.default_rng(0).permutation(len(meta))
    b = E.per_impression_metrics(meta.iloc[perm].reset_index(drop=True), score[perm])
    pd.testing.assert_frame_equal(a, b)
    # strictly monotone transform of the score changes nothing
    c = E.per_impression_metrics(meta, np.exp(score) * 3 + 1)
    pd.testing.assert_frame_equal(a, c, check_exact=False, atol=1e-12)


def test_eligibility_and_bounds(case):
    meta, score, fast = case
    assert fast.loc[fast["n_clicks"] == 0, ["auc", "mrr", "ndcg@10"]].isna().all().all()
    assert fast.loc[fast["n_cold_clicks"] == 0, "cold_recall@10"].isna().all()
    assert fast.loc[fast["n_cold_negatives"] == 0, "cold_auc_within"].isna().all()
    for c in E.metric_columns(fast.reset_index()):
        if c.startswith("novelty"):
            continue
        v = fast[c].dropna()
        assert ((v >= -1e-12) & (v <= 1 + 1e-12)).all(), c


def test_perfect_and_reversed_scores():
    meta, _ = make_meta(seed=5)
    y = meta["label"].to_numpy().astype(float)
    good = E.evaluate(meta, y + 1e-6 * np.arange(len(y)))
    assert good["auc"] == pytest.approx(1.0) and good["ndcg@10"] == pytest.approx(1.0)
    assert good["cold_auc_within"] == pytest.approx(1.0)
    bad = E.evaluate(meta, -y)
    assert bad["auc"] == pytest.approx(0.0)
    const = E.evaluate(meta, np.zeros(len(y)))
    assert const["auc"] == pytest.approx(0.5) and const["cold_auc_within"] == pytest.approx(0.5)


def test_random_expectation_matches_monte_carlo():
    meta, _ = make_meta(seed=6)
    rng = np.random.default_rng(1)
    sims = [E.evaluate(meta, rng.random(len(meta)), extended=False)["cold_recall@10"] for _ in range(300)]
    expected = E.summarise(E.per_impression_metrics(meta, np.zeros(len(meta))))["cold_recall@10_random"]
    assert np.mean(sims) == pytest.approx(expected, abs=0.01)


def test_cluster_bootstrap():
    meta, score = make_meta(seed=7)
    a = E.per_impression_metrics(meta, score)
    b = E.per_impression_metrics(meta, score + 0.3 * meta["label"].to_numpy())
    res = E.cluster_bootstrap(a, b, ["auc", "cold_recall@10"], n_boot=300)
    assert res.loc["auc", "estimate"] == pytest.approx((b["auc"] - a["auc"]).mean())
    assert res.loc["auc", "ci_low"] <= res.loc["auc", "estimate"] <= res.loc["auc", "ci_high"]
    assert res.loc["auc", "n_clusters"] == a.loc[a["auc"].notna(), "user_id"].nunique()
    # identical models: zero-width interval
    z = E.cluster_bootstrap(a, a, ["auc"], n_boot=50)
    assert z.loc["auc", "ci_low"] == z.loc["auc", "ci_high"] == 0
    # clustering by user is at least as wide as resampling impressions when users repeat
    imp = E.cluster_bootstrap(a, b, ["auc"], cluster=None, n_boot=2000, seed=3)
    usr = E.cluster_bootstrap(a, b, ["auc"], cluster="user_id", n_boot=2000, seed=3)
    assert usr.loc["auc", "se"] > 0.5 * imp.loc["auc", "se"]
    # legacy API still works
    p = E.paired_bootstrap(a["auc"], b["auc"], n_boot=200, clusters=a["user_id"])
    assert set(p) == {"mean_diff", "ci_low", "ci_high", "n_impressions"}


def test_segments():
    meta, score = make_meta(seed=8)
    pi = E.per_impression_metrics(meta, score)
    tab = SG.evaluate_by(meta, score, by="exposure")
    assert {"eligible_impressions", "clicks"} <= set(tab.columns)
    assert tab["clicks"].sum() == meta["label"].sum()
    assert tab["slot_share@10"].sum() == pytest.approx(1.0)
    # the union of buckets below 10 reproduces cold recall when computed as one segment
    lab = np.where(meta["prior_exposures"] < 10, "cold", "warm")
    seg = SG.evaluate_by(meta.assign(cw=lab), score, by="cw")
    assert seg.loc["cold", "recall@10"] == pytest.approx(pi["cold_recall@10"].mean())
    assert seg.loc["cold", "auc_vs_all"] == pytest.approx(pi["cold_auc_vs_all"].mean())
    hist = SG.evaluate_by(meta, score, by="history")
    assert hist["eligible_impressions"].sum() == len(pi)


def test_slot_quota_rerank():
    meta, score = make_meta(seed=9, max_slate=40)
    cold = E.cold_mask(meta, 10)
    same = D.slot_quota_rerank(meta, score, cold, 0.0)
    # q=0 keeps the order; only AUC-type metrics differ because ties become strict
    rank_cols = ["mrr", "ndcg@5", "ndcg@10", "cold_recall@10", "warm_recall@10", "cold_slot_share@10"]
    pd.testing.assert_frame_equal(E.per_impression_metrics(meta, same)[rank_cols],
                                  E.per_impression_metrics(meta, score)[rank_cols])
    new = D.slot_quota_rerank(meta, score, cold, 0.3)
    after = E.per_impression_metrics(meta, new).set_index("impression_id")
    before = E.per_impression_metrics(meta, score).set_index("impression_id")
    n_cold = pd.Series(cold).groupby(meta["impression_id"].to_numpy()).sum()
    big = after["n_candidates"] > 10
    need = np.minimum(3, n_cold.reindex(after.index))
    assert (after.loc[big, "cold_slot_share@10"] * 10 >= need[big] - 1e-9).all()
    assert (after["cold_slot_share@10"] >= before["cold_slot_share@10"] - 1e-12).all()


def test_stress_slates():
    meta, _ = make_meta(seed=10, max_slate=40)
    cold = E.cold_mask(meta, 10)
    mask, info = D.stress_slate_mask(meta, cold, slate_size=10, warm_share=0.4, seed=0)
    kept = meta[mask].assign(cold=cold[mask])
    g = kept.groupby("impression_id")
    assert (g.size() == 10).all() and (g["cold"].sum() == 6).all()
    clicked = meta[meta["label"] == 1]
    kept_imps = set(kept["impression_id"])
    assert clicked[clicked["impression_id"].isin(kept_imps)]["row_id"].isin(kept["row_id"]).all()
    assert info["impressions_kept"] == len(kept_imps)


def test_audit_recompute_and_checks():
    meta, score = make_meta(seed=11)
    m = meta.assign(split="val")
    pe = A.recompute_prior_exposures(m)
    brute_pe = [((m["news_id"] == r.news_id) & (m["time"] < r.time)).sum() for r in m.itertuples()]
    np.testing.assert_array_equal(pe, brute_pe)
    pred = pd.DataFrame({"row_id": m["row_id"], "predicted_score": score})
    assert A.check_predictions(m, pred)["ok"]
    bad = pd.concat([pred.iloc[1:], pred.iloc[:1].assign(predicted_score=np.nan)])
    assert not A.check_predictions(m, bad)["ok"]
    np.testing.assert_array_equal(A.align_scores(m, pred.sample(frac=1, random_state=0)), score)


def test_error_cases_and_business():
    meta, score = make_meta(seed=12, max_slate=30)
    cold = E.cold_mask(meta, 10)
    cases = D.error_cases(meta, score, cold)
    assert set(cases["case"]) <= set(D.CASES)
    assert (cases.loc[cases["case"] == "success", "rank"] <= 3).all()
    a = E.per_impression_metrics(meta, score)
    b = E.per_impression_metrics(meta, D.slot_quota_rerank(meta, score, cold, 0.3))
    bi = D.business_impact(a, b, n_boot=100)
    assert bi.loc["n_cold_hits@10", "delta_per_1000"] >= 0


def test_exposure_concentration_and_patterns():
    meta, score = make_meta(seed=13, max_slate=30)
    conc = D.exposure_concentration(meta, {"m": score, "pop": meta["prior_exposures"].to_numpy(float)})
    assert np.allclose(conc.sum(), 1.0)
    groups = D.popularity_groups(meta)
    assert set(groups) <= {"top", "middle", "bottom"}
    cases = D.error_cases(meta, score, E.cold_mask(meta, 10))
    pat = D.tier_patterns(cases)
    assert {"tier", "attribute", "std_diff", "reading"} <= set(pat.columns)
    assert D.direction(0.1, 0.2) == "higher" and D.direction(-0.2, -0.1) == "lower" and D.direction(-0.1, 0.1) == "no detectable change"


def test_tier_contributions_flip_sign():
    abl = pd.DataFrame([{"split": "val", "against": "full", "variant": "no_t2", "metric": "cold_recall@10",
                         "delta": -0.05, "ci_low": -0.08, "ci_high": -0.02}])
    c = D.tier_contributions(abl, "val")
    assert c.loc["no_t2", "cold_recall@10 contribution"] == pytest.approx(0.05)
    assert c.loc["no_t2", "cold_recall@10 ci_low"] == pytest.approx(0.02)
    assert c.loc["no_t2", "cold_recall@10 reading"] == "helps"
    assert c.loc["no_t2", "group removed"] == "T2"
