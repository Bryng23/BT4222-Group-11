<!--
Report sections owned by Shi Ting. Step 6 fills every double-brace placeholder from report_numbers.json and writes
step6_outputs/real/report_sections_filled.md. Unfilled keys show as [[MISSING ...]] (a prediction file was not found).

* Numbers and verdict words ("helps", "no detectable effect", "better for cold articles") are filled automatically.
* [INTERPRET] marks a sentence whose wording depends on the result. Keep the version the numbers support and delete the rest.
* Table / figure names in backticks are files in step6_outputs/real/tables and /figures.
* Main model: deep hybrid (step 4). References: best simple baseline and Group 10's NRMS. Ablation: step 5 variants.
* Section 2.6 (attention read-out) and the training details come from the step 4 and step 5 notebooks, not from step 6.

Structure:
  0 Summary  ->  1 Framework  ->  2 Results layer by layer  ->  3 Ablation  ->  4 Segments
  ->  5 Error analysis  ->  6 Answer to the research question  ->  7 Business insights  ->  8 Limitations
-->

# 0. Summary of results and final business insights

> **Read this first.** Numbers are from the **test** day (14 November 2019; {{support.test.t10.cold_click_impressions}} impressions with a clicked cold article), scored once after every choice was frozen on validation; validation values are given in brackets where they matter. "Cold" means fewer than 10 prior exposures. Intervals are 95%, paired (same impressions, two models) and resample readers. A difference counts only when its interval excludes zero. "Per 1,000 visits" means per 1,000 impressions.

## 0.1 What each step produced

| Step | What it did | Outcome |
|---|---|---|
| 1–2 Data and features | MIND-small train log (156,965 impressions) split by time into train 95,071 impressions (3.39M candidate rows), validation 31,624 (1.23M) and test 30,270 (1.22M). Features built only from events strictly before each impression | Engineered features in four tiers: T1 own behaviour, T2 borrowed behaviour, T3 pure content, U reader match. {{support.test.t10.cold_click_share}} of test clicks are on cold articles |
| 3a Simple baselines | Random, popularity, category CTR, candidate smoothed CTR, content smoothed CTR; smoothing strengths tuned on validation (β = 5, α = 20) | Best simple baseline (validation nDCG@10): {{setting.best_baseline}}, test nDCG@10 {{test.bestbase.ndcg@10}} |
| 3b Group 10 NRMS | Group 10's attention network (past-project baseline, content and history only), run as built on our split; best epoch 10 | Test nDCG@10 {{test.nrms_g10.ndcg@10}}, Cold Recall@10 {{test.nrms_g10.cold_recall@10}} |
| **4 Deep hybrid (main model)** | Group 10's attention network plus an engineered-feature tower (T1/T2/T3/U), candidate-aware attention over the reader's history, a GMF term and an article ID embedding with ID masking (p = 0.3). Best validation epoch 3; trained in 19 minutes on CPU | **Test nDCG@10 {{test.deep_hybrid.ndcg@10}}, Cold Recall@10 {{test.deep_hybrid.cold_recall@10}}** |
| 5 Ablation | Seven variants of the main model, each retrained from scratch with one component removed or added | Section 0.3 |
| 6 Evaluation | Six layers, ablation, segments, error analysis, business translation; nothing trained | This report |

## 0.2 Results by layer

**Headline table (test, top 10 = first screen):**

| Model | AUC | nDCG@10 | Cold Recall@10 | Warm Recall@10 | Never-shown Recall@10 | Within-cold AUC |
|---|---|---|---|---|---|---|
| **Deep hybrid (main)** | **{{test.deep_hybrid.auc}}** | **{{test.deep_hybrid.ndcg@10}}** | **{{test.deep_hybrid.cold_recall@10}}** | **{{test.deep_hybrid.warm_recall@10}}** | **{{test.deep_hybrid.zero_recall@10}}** | **{{test.deep_hybrid.cold_auc_within}}** |
| Group 10 NRMS | {{test.nrms_g10.auc}} | {{test.nrms_g10.ndcg@10}} | {{test.nrms_g10.cold_recall@10}} | {{test.nrms_g10.warm_recall@10}} | {{test.nrms_g10.zero_recall@10}} | {{test.nrms_g10.cold_auc_within}} |
| Best simple baseline ({{setting.best_baseline}}) | {{test.bestbase.auc}} | {{test.bestbase.ndcg@10}} | {{test.bestbase.cold_recall@10}} | {{test.bestbase.warm_recall@10}} | {{test.bestbase.zero_recall@10}} | {{test.bestbase.cold_auc_within}} |
| Content smoothed CTR | {{test.content_smoothed_ctr.auc}} | {{test.content_smoothed_ctr.ndcg@10}} | {{test.content_smoothed_ctr.cold_recall@10}} | {{test.content_smoothed_ctr.warm_recall@10}} | {{test.content_smoothed_ctr.zero_recall@10}} | {{test.content_smoothed_ctr.cold_auc_within}} |
| Category CTR | {{test.category_ctr.auc}} | {{test.category_ctr.ndcg@10}} | {{test.category_ctr.cold_recall@10}} | {{test.category_ctr.warm_recall@10}} | {{test.category_ctr.zero_recall@10}} | {{test.category_ctr.cold_auc_within}} |
| Popularity | {{test.popularity.auc}} | {{test.popularity.ndcg@10}} | {{test.popularity.cold_recall@10}} | {{test.popularity.warm_recall@10}} | {{test.popularity.zero_recall@10}} | {{test.popularity.cold_auc_within}} |
| Random | {{test.random.auc}} | {{test.random.ndcg@10}} | {{test.random.cold_recall@10}} | {{test.random.warm_recall@10}} | {{test.random.zero_recall@10}} | {{test.random.cold_auc_within}} |
| *Validation:* deep hybrid | {{val.deep_hybrid.auc}} | {{val.deep_hybrid.ndcg@10}} | {{val.deep_hybrid.cold_recall@10}} | {{val.deep_hybrid.warm_recall@10}} | {{val.deep_hybrid.zero_recall@10}} | {{val.deep_hybrid.cold_auc_within}} |

**Results by layer.** "References" are the best simple baseline and Group 10's NRMS. "Difference" is main model minus reference, with its 95% interval.

| Layer | Question it answers | Metric (test) | Main model | References | Difference [95% interval] | Verdict | In plain English |
|---|---|---|---|---|---|---|---|
| 0 Trust | Are there enough cold clicks to measure? | Impressions with a clicked cold article | {{support.test.t10.cold_click_impressions}} ({{support.test.t10.cold_clicks}} cold clicks) | {{support.test.t5.cold_click_impressions}} at threshold 5; {{support.test.t20.cold_click_impressions}} at threshold 20 | Margin on Cold Recall ≈ ±{{support.test.t10.ci_halfwidth}} | Enough for medium effects | New-article clicks are rare ({{support.test.t10.cold_click_share}} of all clicks), but there are over 800 of them. That is enough to see a difference of about 3 in 100, not 1 in 100 |
| 0 Trust | Are the cold labels right? | Agreement with an independent recount of prior exposures | Table `audit_cold_flags` | n/a | | [INTERPRET: labels trusted if 100%] | We counted each article's past views a second way and compared it with the stored labels |
| 1 Overall | Is it a good ranker in general? | nDCG@10 | {{test.deep_hybrid.ndcg@10}} | {{test.bestbase.ndcg@10}}; NRMS {{test.nrms_g10.ndcg@10}} | {{rq.vs_bestbase.ndcg@10}} {{rq.vs_bestbase.ndcg@10.ci}}; vs NRMS {{rq.vs_nrms_g10.ndcg@10}} {{rq.vs_nrms_g10.ndcg@10.ci}} | [INTERPRET] | Clicked articles sit nearer the top of the list than under the simple ranker or Group 10's network |
| 1 Overall | Same, ignoring position | AUC | {{test.deep_hybrid.auc}} | {{test.bestbase.auc}}; NRMS {{test.nrms_g10.auc}} | {{test.delta.deep_hybrid_vs_candidate_smoothed_ctr.auc}} {{test.delta.deep_hybrid_vs_candidate_smoothed_ctr.auc.ci}}; vs NRMS {{test.delta.deep_hybrid_vs_nrms_g10.auc}} {{test.delta.deep_hybrid_vs_nrms_g10.auc.ci}} | [INTERPRET] | Pick a clicked and an unclicked article from the same page: we rank the clicked one higher {{test.deep_hybrid.auc.per100}} times in 100, against {{test.bestbase.auc.per100}} for the simple ranker and {{test.nrms_g10.auc.per100}} for NRMS (50 = coin flip) |
| 2 Cold vs warm | Are clicked **new** articles on the first screen? | Cold Recall@10 | {{test.deep_hybrid.cold_recall@10}} | {{test.bestbase.cold_recall@10}}; NRMS {{test.nrms_g10.cold_recall@10}} | {{rq.vs_bestbase.cold_recall@10}} {{rq.vs_bestbase.cold_recall@10.ci}}; vs NRMS {{rq.vs_nrms_g10.cold_recall@10}} {{rq.vs_nrms_g10.cold_recall@10.ci}} | vs baseline: {{rq.vs_bestbase.verdict}}; vs NRMS: {{rq.vs_nrms_g10.verdict}} | Of every 100 new articles readers clicked, {{test.deep_hybrid.cold_recall@10.per100}} were on our first screen, against {{test.bestbase.cold_recall@10.per100}} for the simple ranker and {{test.nrms_g10.cold_recall@10.per100}} for NRMS |
| 2 Cold vs warm | ...without losing **established** ones? | Warm Recall@10 | {{test.deep_hybrid.warm_recall@10}} | {{test.bestbase.warm_recall@10}}; NRMS {{test.nrms_g10.warm_recall@10}} | {{rq.vs_bestbase.warm_recall@10}} {{rq.vs_bestbase.warm_recall@10.ci}}; vs NRMS {{rq.vs_nrms_g10.warm_recall@10}} {{rq.vs_nrms_g10.warm_recall@10.ci}} | (see verdict above) | {{test.deep_hybrid.warm_recall@10.per100}} in 100 clicked established articles were on the first screen, against {{test.bestbase.warm_recall@10.per100}} and {{test.nrms_g10.warm_recall@10.per100}} |
| 2 Cold vs warm | And articles **never shown before**? | Never-shown Recall@10 | {{test.deep_hybrid.zero_recall@10}} | {{test.bestbase.zero_recall@10}}; NRMS {{test.nrms_g10.zero_recall@10}}; random {{test.random.zero_recall@10}} | {{rq.vs_bestbase.zero_recall@10}} {{rq.vs_bestbase.zero_recall@10.ci}}; vs NRMS {{rq.vs_nrms_g10.zero_recall@10}} {{rq.vs_nrms_g10.zero_recall@10.ci}} | [INTERPRET] | For brand-new articles with no views at all: {{test.deep_hybrid.zero_recall@10.per100}} in 100 clicked ones reached the first screen, against {{test.bestbase.zero_recall@10.per100}} for the simple ranker. Few such impressions, so treat as indicative |
| 3 Popularity-neutral | Does it pick the **right** new articles, or push all of them up? | Within-cold AUC | {{test.deep_hybrid.cold_auc_within}} | {{test.bestbase.cold_auc_within}}; NRMS {{test.nrms_g10.cold_auc_within}} | {{rq.vs_bestbase.cold_auc_within}} {{rq.vs_bestbase.cold_auc_within.ci}}; vs NRMS {{rq.vs_nrms_g10.cold_auc_within}} {{rq.vs_nrms_g10.cold_auc_within.ci}} | [INTERPRET] | Given two new articles, one clicked and one not, we rank the clicked one higher {{test.deep_hybrid.cold_auc_within.per100}} times in 100 (simple ranker {{test.bestbase.cold_auc_within.per100}}, NRMS {{test.nrms_g10.cold_auc_within.per100}}, coin flip 50) |
| 3 Popularity-neutral | Does it still win when popular clicks count less? | Popularity-debiased nDCG@10 (γ = 1) | {{test.deep_hybrid.popdebiased_ndcg@10_g1}} | {{test.bestbase.popdebiased_ndcg@10_g1}}; NRMS {{test.nrms_g10.popdebiased_ndcg@10_g1}} | Table `metric_robustness` | [INTERPRET] | Even when a click on a hugely popular article counts less than a click on an obscure one, does our ranker still come out ahead? |
| 4 Exposure | Do new articles get a fair share of the first screen? | Cold exposure ratio (slot share ÷ candidate share; 1 = proportional) | {{test.deep_hybrid.cold_exposure_ratio@10}} | {{test.bestbase.cold_exposure_ratio@10}}; popularity {{test.popularity.cold_exposure_ratio@10}}; random {{test.random.cold_exposure_ratio@10}} | Descriptive (no interval) | [INTERPRET: above / below fair share] | Above 1 means new articles get more first-screen room than their share of the candidate list |
| 4 Exposure | Does the long tail of articles get seen? | Share of first-screen slots for the least-shown 80% of articles | {{conc.test.deep_hybrid.bottom}} | {{conc.test.bestbase.bottom}}; NRMS {{conc.test.nrms_g10.bottom}}; random top 10 {{conc.test.random.bottom}}; popularity {{conc.test.popularity.bottom}} | Descriptive | [INTERPRET] | The random top 10 is the like-for-like reference: MSN's candidate lists are already dominated by a few hits |
| 5 Balance | What do extra slots for new articles cost? | Cold / Warm Recall@10 with 1 slot of 10 reserved (validation) | {{balance.quota10.cold}} / {{balance.quota10.warm}} | No quota: {{balance.quota0.cold}} / {{balance.quota0.warm}} | Per 1,000 test visits: {{business.test.quota10.n_cold_hits@10}} new, {{business.test.quota10.n_warm_hits@10}} established | [INTERPRET: cheap / costly] | Reserving one slot changes new articles found from {{balance.quota0.cold.per100}} to {{balance.quota10.cold.per100}} in 100, and established ones from {{balance.quota0.warm.per100}} to {{balance.quota10.warm.per100}} |
| 5 Balance | Same with 3 slots, and what the rule picks | Cold / Warm Recall@10 with 3 slots of 10 reserved (validation) | {{balance.quota30.cold}} / {{balance.quota30.warm}} | No quota: {{balance.quota0.cold}} / {{balance.quota0.warm}} | Per 1,000 test visits: {{business.test.quota30.n_cold_hits@10}} new, {{business.test.quota30.n_warm_hits@10}} established | Rule picks: {{balance.x1.selected}} (1%), {{balance.x2.selected}} (2%), {{balance.x5.selected}} (5%) | With three slots, {{balance.quota30.cold.per100}} in 100 new articles are found, and established articles go from {{balance.quota0.warm.per100}} to {{balance.quota30.warm.per100}} in 100 |

Threshold check: Table `threshold_survival` says whether the cold result holds at thresholds 5 and 20.

## 0.3 Ablation: which part of the model is responsible?

Contribution = main model minus the variant without that component (positive = the component helps), on test. "Helps" only when the interval excludes zero. Every variant keeps the attention news and user encoders.

| Component removed | nDCG@10 | Cold Recall@10 | Reading (cold) | Warm Recall@10 | Reading (warm) | Never-shown Recall@10 | Within-cold AUC | Matters most at (prior exposures) | Decision rule (cold) |
|---|---|---|---|---|---|---|---|---|---|
| Candidate-aware attention | {{rq.tier.CA.ndcg@10}} {{rq.tier.CA.ndcg@10.ci}} | {{rq.tier.CA.cold_recall@10}} {{rq.tier.CA.cold_recall@10.ci}} | {{rq.tier.CA.cold_recall@10.reading}} | {{rq.tier.CA.warm_recall@10}} {{rq.tier.CA.warm_recall@10.ci}} | {{rq.tier.CA.warm_recall@10.reading}} | {{rq.tier.CA.zero_recall@10}} | {{rq.tier.CA.cold_auc_within}} | {{rq.tier.CA.peak_bucket}} | {{ablation.no_candidate_attention.decision}} |
| ID masking | {{rq.tier.IDM.ndcg@10}} {{rq.tier.IDM.ndcg@10.ci}} | {{rq.tier.IDM.cold_recall@10}} {{rq.tier.IDM.cold_recall@10.ci}} | {{rq.tier.IDM.cold_recall@10.reading}} | {{rq.tier.IDM.warm_recall@10}} {{rq.tier.IDM.warm_recall@10.ci}} | {{rq.tier.IDM.warm_recall@10.reading}} | {{rq.tier.IDM.zero_recall@10}} | {{rq.tier.IDM.cold_auc_within}} | {{rq.tier.IDM.peak_bucket}} | {{ablation.no_id_masking.decision}} |
| T1 own behaviour | {{rq.tier.T1.ndcg@10}} {{rq.tier.T1.ndcg@10.ci}} | {{rq.tier.T1.cold_recall@10}} {{rq.tier.T1.cold_recall@10.ci}} | {{rq.tier.T1.cold_recall@10.reading}} | {{rq.tier.T1.warm_recall@10}} {{rq.tier.T1.warm_recall@10.ci}} | {{rq.tier.T1.warm_recall@10.reading}} | {{rq.tier.T1.zero_recall@10}} | {{rq.tier.T1.cold_auc_within}} | {{rq.tier.T1.peak_bucket}} | {{ablation.no_t1.decision}} |
| T2 borrowed behaviour | {{rq.tier.T2.ndcg@10}} {{rq.tier.T2.ndcg@10.ci}} | {{rq.tier.T2.cold_recall@10}} {{rq.tier.T2.cold_recall@10.ci}} | {{rq.tier.T2.cold_recall@10.reading}} | {{rq.tier.T2.warm_recall@10}} {{rq.tier.T2.warm_recall@10.ci}} | {{rq.tier.T2.warm_recall@10.reading}} | {{rq.tier.T2.zero_recall@10}} | {{rq.tier.T2.cold_auc_within}} | {{rq.tier.T2.peak_bucket}} | {{ablation.no_t2.decision}} |
| T3 explicit content columns | {{rq.tier.T3.ndcg@10}} {{rq.tier.T3.ndcg@10.ci}} | {{rq.tier.T3.cold_recall@10}} {{rq.tier.T3.cold_recall@10.ci}} | {{rq.tier.T3.cold_recall@10.reading}} | {{rq.tier.T3.warm_recall@10}} {{rq.tier.T3.warm_recall@10.ci}} | {{rq.tier.T3.warm_recall@10.reading}} | {{rq.tier.T3.zero_recall@10}} | {{rq.tier.T3.cold_auc_within}} | {{rq.tier.T3.peak_bucket}} | {{ablation.no_t3.decision}} |
| U reader match | {{rq.tier.U.ndcg@10}} {{rq.tier.U.ndcg@10.ci}} | {{rq.tier.U.cold_recall@10}} {{rq.tier.U.cold_recall@10.ci}} | {{rq.tier.U.cold_recall@10.reading}} | {{rq.tier.U.warm_recall@10}} {{rq.tier.U.warm_recall@10.ci}} | {{rq.tier.U.warm_recall@10.reading}} | {{rq.tier.U.zero_recall@10}} | {{rq.tier.U.cold_auc_within}} | {{rq.tier.U.peak_bucket}} | {{ablation.no_u.decision}} |

Adding content smoothed CTR as a feature (`full_plus_smoothing`) changes test Cold Recall@10 by {{ablation.test.full_plus_smoothing_vs_full.cold_recall@10}} {{ablation.test.full_plus_smoothing_vs_full.cold_recall@10.ci}} and nDCG@10 by {{ablation.test.full_plus_smoothing_vs_full.ndcg@10}} {{ablation.test.full_plus_smoothing_vs_full.ndcg@10.ci}}.

What the ablation shows. These readings come from the step 5 notebook (impression-level intervals); [INTERPRET] check each one against the table above:

* **New articles are placed by the network that reads them and the reader, not by the engineered tiers.** Removing any single engineered tier left test Cold Recall@10 unchanged within its interval. The content and history encoders are in every variant. Group 10's content-only network alone reaches {{test.nrms_g10.cold_recall@10}} test Cold Recall@10.
* **Candidate-aware attention is the one component that clearly helps cold articles on both splits.** Its attention weights stayed uniform (section 2.6), so what it adds is a second, candidate-specific match between the reader's average history and the article, not the selection of particular past clicks.
* **T1 carries the established-article ranking.** Removing it gives the largest drops in nDCG@10 and Warm Recall@10, and no detectable cold effect, as section 1.4 expects.
* **U (explicit reader–article match) is the second pillar overall.** Its cold contribution was large on validation and not confirmed on test.
* **ID masking works as a regulariser, not as a cold-start device.** It improves overall nDCG@10 but not cold recall. Without masking, the network peaked at epoch 1 and then overfitted.
* **T2 and the smoothed-CTR feature add nothing detectable** once the rest of the model is present.

## 0.4 Segments and errors

* **By article exposure:** for never-shown articles the main model reaches Recall@10 {{test.seg.exposure.0.deep_hybrid.recall@10}}, against {{test.seg.exposure.0.bestbase.recall@10}} for the best simple baseline and {{test.seg.exposure.0.nrms_g10.recall@10}} for NRMS (Figure `fig_recall_by_exposure`).
* **By reader history:** for readers with 50+ past clicks, Cold Recall@10 is {{test.seg.history.50+.deep_hybrid.cold_recall@10}} vs {{test.seg.history.50+.bestbase.cold_recall@10}}; for readers with no history it is {{test.seg.history.0.deep_hybrid.cold_recall@10}} vs {{test.seg.history.0.bestbase.cold_recall@10}} (Table `segment_history`; check its support). [INTERPRET] If the advantage grows with history, matching to the reader is doing the work.
* **By category:** {{business.test.n_categories_gain}} of {{business.test.n_categories}} categories gain detectably in Cold Recall@10 and {{business.test.n_categories_loss}} lose. The largest gain is in {{business.test.top_category}} ({{business.test.top_category.gain.per100}} points, {{business.test.top_category.ci}}).
* **Errors (validation):** {{error.success.count}} clicked cold articles were ranked in the top 3 and {{error.failure.count}} in the bottom quarter. The largest difference between failures and successes: {{error.pattern1}}.

## 0.5 Answer to the research question (test)

**Part 1: does combining content, behaviour and reader evidence help new articles?** Against the best simple baseline the main model is **{{rq.vs_bestbase.verdict}}**. Of every 100 clicked new articles, it put {{test.deep_hybrid.cold_recall@10.per100}} on the first screen, against {{test.bestbase.cold_recall@10.per100}} for the best simple baseline and {{test.nrms_g10.cold_recall@10.per100}} for Group 10's content-only network. For established articles: {{test.deep_hybrid.warm_recall@10.per100}} vs {{test.bestbase.warm_recall@10.per100}} and {{test.nrms_g10.warm_recall@10.per100}}. Against NRMS it is **{{rq.vs_nrms_g10.verdict}}**.

**Part 2: the evidence does different jobs.** T1 (own behaviour) on cold articles: **{{rq.tier.T1.cold_recall@10.reading}}**, on warm: **{{rq.tier.T1.warm_recall@10.reading}}**. U (reader match) on cold: **{{rq.tier.U.cold_recall@10.reading}}**, on warm: **{{rq.tier.U.warm_recall@10.reading}}**. Candidate-aware match on cold: **{{rq.tier.CA.cold_recall@10.reading}}**. T2 on cold: **{{rq.tier.T2.cold_recall@10.reading}}**; T3 on cold: **{{rq.tier.T3.cold_recall@10.reading}}**.

## 0.6 Final business insights

These translate the test results for a news editor or product manager. All are offline estimates: we re-ordered slates MSN had already chosen and counted where the articles readers did click would have appeared.

**1. New articles can be surfaced before they have a click history, without hurting established ones.** Per 1,000 visits, the main model puts **{{business.test.deep_hybrid_vs_bestbase.n_cold_hits@10}} more wanted new articles** on the first screen than the best simple ranker {{business.test.deep_hybrid_vs_bestbase.n_cold_hits@10.ci}}, and changes wanted established articles by {{business.test.deep_hybrid_vs_bestbase.n_warm_hits@10}} {{business.test.deep_hybrid_vs_bestbase.n_warm_hits@10.ci}}. Ranking by popularity, the common default, finds only {{test.popularity.cold_recall@10.per100}} in 100 wanted new articles, against {{test.random.cold_recall@10.per100}} for random.

**2. Reading the article and the reader is what makes new articles rankable; the article's own counters keep established ones ranked well.** The content-only network (NRMS) finds {{test.nrms_g10.cold_recall@10.per100}} of 100 new articles but only {{test.nrms_g10.warm_recall@10.per100}} established ones. The behaviour-only baseline is the reverse ({{test.bestbase.cold_recall@10.per100}} and {{test.bestbase.warm_recall@10.per100}}). The main model combines both: {{test.deep_hybrid.cold_recall@10.per100}} new and {{test.deep_hybrid.warm_recall@10.per100}} established. Against NRMS that is {{business.test.deep_hybrid_vs_nrms_g10.n_cold_hits@10}} wanted new articles {{business.test.deep_hybrid_vs_nrms_g10.n_cold_hits@10.ci}} and {{business.test.deep_hybrid_vs_nrms_g10.n_warm_hits@10}} established ones {{business.test.deep_hybrid_vs_nrms_g10.n_warm_hits@10.ci}} per 1,000 visits.

**3. Invest in article metadata, reading histories and real-time article counters; borrowed category click feeds are optional.** Removing borrowed category and topic click rates (T2) gives {{rq.tier.T2.cold_recall@10.reading}} on cold articles and {{rq.tier.T2.ndcg@10.reading}} overall. The engineered reader–article match (U) and the article's own counters (T1) are the largest contributors overall (nDCG@10 {{rq.tier.U.ndcg@10}} and {{rq.tier.T1.ndcg@10}}). For a newsroom, titles, abstracts, entity tags, clean reading histories and live click counters matter more than streaming category-level CTR pipelines.

**4. There is a dial, and its price is known.** Reserving first-screen slots for the best-scoring new articles, per 1,000 test visits:

| Slots reserved | Extra wanted new articles | Change in wanted established articles |
|---|---|---|
| 1 of 10 | {{business.test.quota10.n_cold_hits@10}} | {{business.test.quota10.n_warm_hits@10}} |
| 2 of 10 | {{business.test.quota20.n_cold_hits@10}} | {{business.test.quota20.n_warm_hits@10}} |
| 3 of 10 | {{business.test.quota30.n_cold_hits@10}} | {{business.test.quota30.n_warm_hits@10}} |

Against about {{business.test.deep_hybrid.n_warm_hits@10.level}} wanted established articles shown per 1,000 visits, these losses are small. Under the balance rule (warm Recall@10 within 2% of the best baseline's), the chosen setting is **{{balance.x2.selected}}**. [INTERPRET] Recommend 1–2 slots as a starting point if the rule does not bind.

**5. Fresh is not the same as the long tail.** The model gives new articles {{test.deep_hybrid.cold_exposure_ratio@10}} times their share of candidates on the first screen. The least-shown 80% of articles get {{conc.test.deep_hybrid.bottom}} of first-screen slots, against {{conc.test.random.bottom}} for a random top 10. A business that cares about creator or publisher fairness should track both KPIs separately.

**What to do next.** Run an online A/B test of the main model with a 1-slot quota for new articles, measuring click-through, time spent, cold slot share and long-tail share against the current ranker.

# 1. Evaluation framework

## 1.1 The problem in one paragraph

Every time a reader opens MSN News, the site shows a list of candidate articles (an *impression*, 2 to 299 articles long). The reader clicks some of them. A ranker's job is to put the articles that will be clicked near the top. The difficulty is that ranking on past clicks favours articles that have already been shown a lot: in our data the top 1% of articles take 32.5% of impressions while the bottom 80% share 2.5%. New, *cold* articles never get the exposure they need to build a click history. We ask whether a ranker that combines several kinds of evidence can find good cold articles **without** pushing down the established, *warm* articles readers also want.

**Words used throughout**

| Term | Meaning |
|---|---|
| Impression / slate | One list of candidate articles shown to one reader at one moment |
| Cold article | Shown fewer than 10 times before this impression (we also check 5 and 20) |
| Zero-exposure article | Never shown before: the hardest cold case |
| Warm article | Shown 10 or more times before |
| First screen / top 10 | The ten highest-ranked articles in a slate |
| Clicked cold article | A cold article the reader actually clicked in that impression |

## 1.2 Four kinds of evidence (T1, T2, T3, U) and how the main model reads them

| Tier | Plain description | Example features | Available for a cold article? | Available for a new reader? |
|---|---|---|---|---|
| **T1** own behaviour | How this article has performed so far | prior clicks and exposures, prior CTR, clicks in the last 6 hours, age | **Weak**: zero or very noisy below 10 exposures | Yes |
| **T2** borrowed behaviour | How *similar* articles are performing right now | recent CTR of the same category, subcategory, entities; CTR of nearest-neighbour articles | **Yes**: borrowed from neighbours | Yes |
| **T3** pure content | What the article is about | category, subcategory, entity count | **Yes**: always | Yes |
| **U** user–article match | How well the article fits this reader | category affinity, text and entity similarity to reading history | Yes, if the reader has history | **Weak**: empty for new readers |

The main model (step 4) reads content and the reader's history directly, through Group 10's attention news and user encoders (title, abstract and WikiData entity vectors). It adds the four tiers as an engineered-feature tower, a candidate-aware attention over the reader's history, a GMF term and an article ID embedding with ID masking. The design idea: T1 is strong for warm articles and useless for cold ones; content and T2 should fill the gap for cold articles; U and the history encoder personalise both.

## 1.3 Six layers of evaluation

No single number can answer the research question. "Is the model good?" splits into six smaller questions, and each needs its own measurement. The layers are ordered: each one only makes sense once the one before it is settled.

| Layer | The question in one sentence | Main measurement | Pass mark |
|---|---|---|---|
| 0 Trust | Can we measure cold-article performance reliably at all? | Number of impressions with a clicked cold article | Enough impressions for a small margin of error |
| 1 Overall | Is the model a competent ranker in general? | AUC, nDCG@10 | At least as good as the best baseline |
| 2 Cold vs warm | When a reader clicked a new article, was it on the first screen, and is the same still true for established articles? | Cold Recall@10 next to Warm Recall@10 | Cold up, warm not down |
| 3 Popularity-neutral | Can the model tell good new articles from bad ones, rather than just pushing all new articles up? | Within-cold AUC | Clearly above 0.5 and above baselines |
| 4 Exposure | Does the model share first-screen slots more fairly between popular articles and the long tail? | Share of first-screen slots for the least-shown 80% of articles | Higher than a random top 10 and the popularity ranker |
| 5 Balance | If we give new articles more slots on purpose, what does it cost, and where should we stop? | Cold-vs-warm trade-off curve | A setting that gains cold within the agreed warm loss |

**Layer 0: Trust.** Every cold metric is averaged only over impressions in which the reader clicked at least one cold article. The margin of error on Cold Recall is roughly ±0.8 / √n: about ±0.025 with 1,000 impressions, ±0.045 with 300. The cold flags are recomputed independently from the raw log. The threshold of 10 is justified by (a) how badly an article's CTR after *n* exposures predicts its later CTR and (b) how many hours an article takes to reach 5, 10 and 20 exposures.

**Layer 1: Overall.** AUC, MRR and nDCG@5/10, the standard MIND metrics. Only about 2% of test clicks are on cold articles, so this layer is almost entirely a measure of **warm** ranking. It is a guard: the model must not give up general quality to help new articles.

**Layer 2: Cold vs warm.** Cold Recall@10 (share of clicked cold articles in the top 10 of the full slate), Warm Recall@10, Zero-exposure Recall@10 and Cold Hit@10. Cold and warm are always reported **side by side**, never blended, so a cold gain cannot hide a warm loss. Each number should be read against a random ranker, which already places min(10, *n*)/*n* of clicked articles in the top 10 of an *n*-article slate.

**Layer 3: Popularity-neutral.** A model that simply ranks every cold article higher will raise Cold Recall. Within-cold AUC compares cold articles only against other cold articles in the same slate (0.5 = cannot tell good from bad). Cold-vs-all AUC asks whether a good new article beats popular ones. Popularity-debiased nDCG@10 counts a click on a rarely-shown article more than a click on a heavily-shown one (weight 1 / (prior exposures + 1)^γ).

**Layer 4: Exposure.** The rich-get-richer loop: an article can only collect clicks, and become warm, if it reaches the first screen. Measured by cold slot share@10, the exposure ratio (cold slot share ÷ cold share of candidates; 1 = proportional), exposure concentration (share of first-screen slots for the most-shown 1% and least-shown 80% of articles), coverage@10 and novelty@10. Any top-10 ranker concentrates exposure more than the logged slates, so the like-for-like reference is a random top 10.

**Layer 5: Balance.** A slot-quota re-ranker on saved scores reserves 0, 10, 20 or 30% of the top 10 for the best-scoring cold articles, with no retraining. The balance rule picks the setting with the highest cold Recall@10 whose warm Recall@10 stays within *x*% of the best baseline (*x* = 1, 2, 5%), on validation.

**Lift over random.** Slates vary from 2 to 299 articles, so whenever groups of slates with different lengths are compared, we report *lift*: the metric minus what a random ranker would score on the same slates.

## 1.4 Linking evidence to layers: what we expect to see

| Component | Should move | Should *not* move much | Tested in |
|---|---|---|---|
| T1 own behaviour | Warm Recall@10 and overall nDCG; exposure buckets 20+ | Zero-exposure recall | Ablation `no_t1` |
| T2 borrowed behaviour | Cold Recall@10, within-cold AUC; exposure buckets 0–9 | Warm articles, where T1 already knows more | Ablation `no_t2` |
| T3 explicit content | Zero-exposure recall, within-cold AUC | Warm articles | Ablation `no_t3` (the news encoder reads content in every variant) |
| U user matching | Every layer; most for readers with history | New readers (history 0) | Ablation `no_u`; user-history segment |
| Candidate-aware attention | Cold Recall@10, by letting each candidate pick the relevant past clicks | n/a | Ablation `no_candidate_attention`; attention read-out (section 2.6) |
| ID masking | Cold Recall@10, by stopping the network from relying on ID look-ups | n/a | Ablation `no_id_masking` |

## 1.5 Rules that apply to every layer

* **Splits.** Train 9–12 November, validation 13 November, test 14 November 2019. Every decision was made on validation; test was read once, at the end.
* **Cold threshold.** Fewer than 10 prior exposures at the moment of the impression (Figure `fig_threshold_justification`):
  * *Noise.* Early CTR is a poor guide to an article's later CTR. Its error falls to {{noise.n5.rel}} of the one-exposure value after 5 exposures, {{noise.n10.rel}} after 10 and {{noise.n20.rel}} after 20.
  * *Time.* The median article reaches 10 exposures {{volume.t10.median_hours}} hours after it is first shown (5 exposures: {{volume.t5.median_hours}} h; 20: {{volume.t20.median_hours}} h). Only {{volume.t10.share_reaching}} of articles ever reach 10.
  * So "cold" covers roughly an article's first {{volume.t10.median_hours}} hours: the window in which its own clicks are still too noisy to rank on.
* **Counters** keep updating through validation and test, as in a live system.
* **Ties** are broken by a label-blind hash, identical for every model.
* **Uncertainty.** Every comparison is paired (same impressions, two models) with a 95% interval that resamples *readers*, because one reader's impressions are related. We say "better" or "worse" only when the interval excludes zero; otherwise "no detectable difference". {{setting.n_boot}} resamples.
* **Support.** Every table shows how many impressions and clicks each number rests on.
* **References.** Category CTR, candidate smoothed CTR, Group 10's NRMS, and the strongest simple baseline on validation ({{setting.best_baseline}}).

# 2. Results, layer by layer

All numbers are on the test day unless marked. Full train / validation / test tables: `headline_all_models`; all differences: `headline_deltas`, Figure `fig_forest_main_vs_references`.

## Layer 0: can we trust the test?

The test day has **{{support.test.t10.cold_click_impressions}}** impressions with at least one clicked cold article ({{support.test.t10.cold_clicks}} cold clicks; validation {{support.val.t10.cold_click_impressions}}). With that many, a Cold Recall difference has a margin of error of roughly ±{{support.test.t10.ci_halfwidth}}. At the stricter threshold of 5 the count falls to {{support.test.t5.cold_click_impressions}}; at 20 it rises to {{support.test.t20.cold_click_impressions}} (Table `support_cold_clicks`). The cold flags were recounted independently (Table `audit_cold_flags`). [INTERPRET] Agreement is complete / differences are explained by …

## Layer 1: overall ranking

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **Deep hybrid (main)** | {{test.deep_hybrid.auc}} | {{test.deep_hybrid.mrr}} | {{test.deep_hybrid.ndcg@5}} | {{test.deep_hybrid.ndcg@10}} |
| Group 10 NRMS | {{test.nrms_g10.auc}} | {{test.nrms_g10.mrr}} | {{test.nrms_g10.ndcg@5}} | {{test.nrms_g10.ndcg@10}} |
| Best simple baseline ({{setting.best_baseline}}) | {{test.bestbase.auc}} | {{test.bestbase.mrr}} | {{test.bestbase.ndcg@5}} | {{test.bestbase.ndcg@10}} |
| Content smoothed CTR | {{test.content_smoothed_ctr.auc}} | {{test.content_smoothed_ctr.mrr}} | {{test.content_smoothed_ctr.ndcg@5}} | {{test.content_smoothed_ctr.ndcg@10}} |
| Category CTR | {{test.category_ctr.auc}} | {{test.category_ctr.mrr}} | {{test.category_ctr.ndcg@5}} | {{test.category_ctr.ndcg@10}} |
| Popularity | {{test.popularity.auc}} | {{test.popularity.mrr}} | {{test.popularity.ndcg@5}} | {{test.popularity.ndcg@10}} |
| Random | {{test.random.auc}} | {{test.random.mrr}} | {{test.random.ndcg@5}} | {{test.random.ndcg@10}} |

Against the best simple baseline the main model changes nDCG@10 by {{test.delta.deep_hybrid_vs_candidate_smoothed_ctr.ndcg@10}} {{test.delta.deep_hybrid_vs_candidate_smoothed_ctr.ndcg@10.ci}}; against NRMS by {{test.delta.deep_hybrid_vs_nrms_g10.ndcg@10}} {{test.delta.deep_hybrid_vs_nrms_g10.ndcg@10.ci}} (validation {{val.delta.deep_hybrid_vs_nrms_g10.ndcg@10}} {{val.delta.deep_hybrid_vs_nrms_g10.ndcg@10.ci}}).

**Training vs test.** Train nDCG@10 is {{train.deep_hybrid.ndcg@10}}, validation {{val.deep_hybrid.ndcg@10}}, test {{test.deep_hybrid.ndcg@10}} (NRMS: {{train.nrms_g10.ndcg@10}}, {{val.nrms_g10.ndcg@10}}, {{test.nrms_g10.ndcg@10}}). Training-split values are in-sample. Early stopping picked epoch 3 of 6; validation nDCG@10 fell after that while training loss kept dropping.

*Reading:* this layer is about 98% warm clicks, so it shows the model is a competent general ranker, not yet anything about cold articles.

## Layer 2: cold vs warm vs never-shown

| Model | Cold Recall@10 | Cold Hit@10 | Warm Recall@10 | Zero-exposure Recall@10 |
|---|---|---|---|---|
| **Deep hybrid (main)** | **{{test.deep_hybrid.cold_recall@10}}** | {{test.deep_hybrid.cold_hit@10}} | {{test.deep_hybrid.warm_recall@10}} | {{test.deep_hybrid.zero_recall@10}} |
| Group 10 NRMS | {{test.nrms_g10.cold_recall@10}} | {{test.nrms_g10.cold_hit@10}} | {{test.nrms_g10.warm_recall@10}} | {{test.nrms_g10.zero_recall@10}} |
| Best simple baseline | {{test.bestbase.cold_recall@10}} | {{test.bestbase.cold_hit@10}} | {{test.bestbase.warm_recall@10}} | {{test.bestbase.zero_recall@10}} |
| Content smoothed CTR | {{test.content_smoothed_ctr.cold_recall@10}} | {{test.content_smoothed_ctr.cold_hit@10}} | {{test.content_smoothed_ctr.warm_recall@10}} | {{test.content_smoothed_ctr.zero_recall@10}} |
| Category CTR | {{test.category_ctr.cold_recall@10}} | {{test.category_ctr.cold_hit@10}} | {{test.category_ctr.warm_recall@10}} | {{test.category_ctr.zero_recall@10}} |
| Popularity | {{test.popularity.cold_recall@10}} | {{test.popularity.cold_hit@10}} | {{test.popularity.warm_recall@10}} | {{test.popularity.zero_recall@10}} |
| Random | {{test.random.cold_recall@10}} | {{test.random.cold_hit@10}} | {{test.random.warm_recall@10}} | {{test.random.zero_recall@10}} |

* **vs the best simple baseline:** cold {{rq.vs_bestbase.cold_recall@10}} {{rq.vs_bestbase.cold_recall@10.ci}}; warm {{rq.vs_bestbase.warm_recall@10}} {{rq.vs_bestbase.warm_recall@10.ci}}. **{{rq.vs_bestbase.verdict}}**.
* **vs Group 10's NRMS:** cold {{rq.vs_nrms_g10.cold_recall@10}} {{rq.vs_nrms_g10.cold_recall@10.ci}} (validation {{val.delta.deep_hybrid_vs_nrms_g10.cold_recall@10}} {{val.delta.deep_hybrid_vs_nrms_g10.cold_recall@10.ci}}); warm {{rq.vs_nrms_g10.warm_recall@10}} {{rq.vs_nrms_g10.warm_recall@10.ci}}. **{{rq.vs_nrms_g10.verdict}}**. The engineered tower did not cost the content network its cold strength.
* **vs category CTR (the baseline best at finding new articles):** cold {{rq.vs_category_ctr.cold_recall@10}} {{rq.vs_category_ctr.cold_recall@10.ci}}. **{{rq.vs_category_ctr.verdict}}**.

*Threshold check* (Table `threshold_survival`): [INTERPRET] the cold-vs-warm conclusion holds at 5, 10 and 20 / holds only at …

*A note on variance.* Validation Cold Recall@10 of the main model swung between 0.60 and 0.72 from epoch to epoch (step 4). Cold differences of 2 to 3 points between deep runs are within training noise.

## Layer 3: is the cold result real?

| Model | Within-cold AUC | Cold-vs-all AUC | Cold-only nDCG@10 | Popularity-debiased nDCG@10 (γ = 1) |
|---|---|---|---|---|
| **Deep hybrid (main)** | {{test.deep_hybrid.cold_auc_within}} | {{test.deep_hybrid.cold_auc_vs_all}} | {{test.deep_hybrid.cold_ndcg@10}} | {{test.deep_hybrid.popdebiased_ndcg@10_g1}} |
| Group 10 NRMS | {{test.nrms_g10.cold_auc_within}} | {{test.nrms_g10.cold_auc_vs_all}} | {{test.nrms_g10.cold_ndcg@10}} | {{test.nrms_g10.popdebiased_ndcg@10_g1}} |
| Best simple baseline | {{test.bestbase.cold_auc_within}} | {{test.bestbase.cold_auc_vs_all}} | {{test.bestbase.cold_ndcg@10}} | {{test.bestbase.popdebiased_ndcg@10_g1}} |
| Content smoothed CTR | {{test.content_smoothed_ctr.cold_auc_within}} | {{test.content_smoothed_ctr.cold_auc_vs_all}} | {{test.content_smoothed_ctr.cold_ndcg@10}} | {{test.content_smoothed_ctr.popdebiased_ndcg@10_g1}} |
| Category CTR | {{test.category_ctr.cold_auc_within}} | {{test.category_ctr.cold_auc_vs_all}} | {{test.category_ctr.cold_ndcg@10}} | {{test.category_ctr.popdebiased_ndcg@10_g1}} |
| Random | {{test.random.cold_auc_within}} | {{test.random.cold_auc_vs_all}} | {{test.random.cold_ndcg@10}} | {{test.random.popdebiased_ndcg@10_g1}} |

Within-cold AUC difference: vs best baseline {{rq.vs_bestbase.cold_auc_within}} {{rq.vs_bestbase.cold_auc_within.ci}}; vs NRMS {{rq.vs_nrms_g10.cold_auc_within}} {{rq.vs_nrms_g10.cold_auc_within.ci}}. Table `metric_robustness` repeats every comparison under each popularity-neutral metric.

[INTERPRET] If layer 2 and layer 3 agree, the cold gain comes from telling good cold articles from bad ones, not from pushing all cold articles up. If layer 2 improves but within-cold AUC does not, the model is promoting cold articles wholesale, which slot quotas can do without a model.

## Layer 4: who gets the first screen?

| Share of exposure going to … | Logged (what MSN showed) | Deep hybrid, top 10 | NRMS, top 10 | Best simple baseline, top 10 | Popularity, top 10 | Random, top 10 |
|---|---|---|---|---|---|---|
| Most-shown 1% of articles | {{conc.test.logged.top}} | {{conc.test.deep_hybrid.top}} | {{conc.test.nrms_g10.top}} | {{conc.test.bestbase.top}} | {{conc.test.popularity.top}} | {{conc.test.random.top}} |
| Least-shown 80% of articles | {{conc.test.logged.bottom}} | {{conc.test.deep_hybrid.bottom}} | {{conc.test.nrms_g10.bottom}} | {{conc.test.bestbase.bottom}} | {{conc.test.popularity.bottom}} | {{conc.test.random.bottom}} |

Cold articles hold {{test.deep_hybrid.cold_slot_share@10}} of the main model's first-screen slots. The exposure ratio (their slot share divided by their share of candidates; 1 = proportional) is {{test.deep_hybrid.cold_exposure_ratio@10}}, against {{test.bestbase.cold_exposure_ratio@10}} for the best simple baseline, {{test.popularity.cold_exposure_ratio@10}} for the popularity ranker and {{test.random.cold_exposure_ratio@10}} for random. Coverage@10: {{test.deep_hybrid.coverage@10}} (random {{test.random.coverage@10}}). Table `exposure_concentration`.

*Reading.* Any top-10 ranker concentrates exposure more than the logged slates, because the hits appear in many slates at once, so the like-for-like reference is the random top 10. [INTERPRET] Compare the main model's long-tail share with random's and the baselines'.

## Layer 5: balance

Figure `fig_tradeoff_val` places every model by warm Recall@10 (x) and cold Recall@10 (y); up and to the right is better on both. The lines show what happens when part of the first screen is reserved for the best-scoring cold articles (no retraining):

| First-screen slots reserved for cold | Cold Recall@10 | Warm Recall@10 |
|---|---|---|
| 0% (model as trained) | {{balance.quota0.cold}} | {{balance.quota0.warm}} |
| 10% (1 of 10) | {{balance.quota10.cold}} | {{balance.quota10.warm}} |
| 20% (2 of 10) | {{balance.quota20.cold}} | {{balance.quota20.warm}} |
| 30% (3 of 10) | {{balance.quota30.cold}} | {{balance.quota30.warm}} |

(Validation.) Under the balance rule "highest cold Recall@10 while warm Recall@10 stays within 2% of the best baseline ({{balance.ref_model}}, {{balance.ref_warm}})", the chosen setting is **{{balance.x2.selected}}** (1%: {{balance.x1.selected}}; 5%: {{balance.x5.selected}}; Table `balance_rule`).

## 2.6 Inside the model: what the attention picks out

From step 4 (section 7), on 19,190 test impressions with at least two history items, with the clicked article as the candidate:

| | Group 10 attention (fixed query) | Candidate-aware attention | Reference |
|---|---|---|---|
| Mean weight on the top history item | 0.164 | 0.133 | Uniform 1/n: 0.133 |
| Top-attended item shares the clicked article's category | 19.7% | 18.3% | Random history item: 20.2% |

Neither attention singles out relevant past clicks. Group 10's attention is only slightly more peaked than uniform; the candidate-aware attention stayed exactly uniform, so in effect it averages the reader's last 10 clicks and matches that average against each candidate. The personalisation the model uses therefore comes from the reader's average taste profile and the engineered matching features (U), not from picking out particular past clicks.

## Summary of chapter 2

| Layer | One-line result (test) |
|---|---|
| 0 Trust | {{support.test.t10.cold_click_impressions}} cold-click impressions; [INTERPRET enough / thin] |
| 1 Overall | nDCG@10 {{test.deep_hybrid.ndcg@10}} vs best baseline {{test.bestbase.ndcg@10}} and NRMS {{test.nrms_g10.ndcg@10}} |
| 2 Cold vs warm | {{rq.vs_bestbase.verdict}} (vs best simple baseline); {{rq.vs_nrms_g10.verdict}} (vs NRMS) |
| 3 Popularity-neutral | Within-cold AUC {{test.deep_hybrid.cold_auc_within}} vs {{test.bestbase.cold_auc_within}}; [INTERPRET agrees / disagrees with layer 2] |
| 4 Exposure | Least-shown 80% get {{conc.test.deep_hybrid.bottom}} of first-screen slots (random top 10: {{conc.test.random.bottom}}; logged: {{conc.test.logged.bottom}}) |
| 5 Balance | Selected setting: {{balance.x2.selected}} |

# 3. Ablation: which evidence is responsible?

Chapter 2 says *what* the main model does. This chapter asks *which component causes it*. Each variant is the main model retrained from scratch with one change, using the same seed, inputs, training loop and best-validation-epoch rule (step 5; Table `ablation_variants`).

| Variant | What changes | Best epoch (step 5) |
|---|---|---|
| `full` | Main model (step 4) | 3 |
| `no_candidate_attention` | Without the candidate-aware attention branch | see step 5 |
| `no_id_masking` | ID vectors never masked in training | 1 |
| `no_t1` | Without the article's own behaviour | 3 |
| `no_t2` | Without borrowed behaviour | 3 |
| `no_t3` | Without explicit content columns (codes, entity counts) | 3 |
| `no_u` | Without the engineered reader–article matching columns | 5 |
| `full_plus_smoothing` | Adds content smoothed CTR (α = 20) | 3 |

**Contribution** of a component = full model minus the model without it. **Positive means it helps.** A contribution counts only if its interval excludes zero.

### Step 1: overall effect (layer 1)

Removing each component changes test nDCG@10 by (variant minus full; negative = it was helping): candidate-aware attention {{ablation.test.no_candidate_attention_vs_full.ndcg@10}} {{ablation.test.no_candidate_attention_vs_full.ndcg@10.ci}}, ID masking {{ablation.test.no_id_masking_vs_full.ndcg@10}} {{ablation.test.no_id_masking_vs_full.ndcg@10.ci}}, T1 {{ablation.test.no_t1_vs_full.ndcg@10}} {{ablation.test.no_t1_vs_full.ndcg@10.ci}}, T2 {{ablation.test.no_t2_vs_full.ndcg@10}} {{ablation.test.no_t2_vs_full.ndcg@10.ci}}, T3 {{ablation.test.no_t3_vs_full.ndcg@10}} {{ablation.test.no_t3_vs_full.ndcg@10.ci}}, U {{ablation.test.no_u_vs_full.ndcg@10}} {{ablation.test.no_u_vs_full.ndcg@10.ci}}.

### Step 2: cold vs warm (layer 2). This is the second half of the research question

| Component | Cold Recall@10, validation (variant − full) | Cold Recall@10, test (contribution) | Reading (test) | Warm Recall@10, test | Reading (warm) | Zero-exposure Recall@10, test |
|---|---|---|---|---|---|---|
| Candidate-aware attention | {{ablation.val.no_candidate_attention_vs_full.cold_recall@10}} {{ablation.val.no_candidate_attention_vs_full.cold_recall@10.ci}} | {{rq.tier.CA.cold_recall@10}} {{rq.tier.CA.cold_recall@10.ci}} | {{rq.tier.CA.cold_recall@10.reading}} | {{rq.tier.CA.warm_recall@10}} {{rq.tier.CA.warm_recall@10.ci}} | {{rq.tier.CA.warm_recall@10.reading}} | {{rq.tier.CA.zero_recall@10}} |
| ID masking | {{ablation.val.no_id_masking_vs_full.cold_recall@10}} {{ablation.val.no_id_masking_vs_full.cold_recall@10.ci}} | {{rq.tier.IDM.cold_recall@10}} {{rq.tier.IDM.cold_recall@10.ci}} | {{rq.tier.IDM.cold_recall@10.reading}} | {{rq.tier.IDM.warm_recall@10}} {{rq.tier.IDM.warm_recall@10.ci}} | {{rq.tier.IDM.warm_recall@10.reading}} | {{rq.tier.IDM.zero_recall@10}} |
| T1 own behaviour | {{ablation.val.no_t1_vs_full.cold_recall@10}} {{ablation.val.no_t1_vs_full.cold_recall@10.ci}} | {{rq.tier.T1.cold_recall@10}} {{rq.tier.T1.cold_recall@10.ci}} | {{rq.tier.T1.cold_recall@10.reading}} | {{rq.tier.T1.warm_recall@10}} {{rq.tier.T1.warm_recall@10.ci}} | {{rq.tier.T1.warm_recall@10.reading}} | {{rq.tier.T1.zero_recall@10}} |
| T2 borrowed behaviour | {{ablation.val.no_t2_vs_full.cold_recall@10}} {{ablation.val.no_t2_vs_full.cold_recall@10.ci}} | {{rq.tier.T2.cold_recall@10}} {{rq.tier.T2.cold_recall@10.ci}} | {{rq.tier.T2.cold_recall@10.reading}} | {{rq.tier.T2.warm_recall@10}} {{rq.tier.T2.warm_recall@10.ci}} | {{rq.tier.T2.warm_recall@10.reading}} | {{rq.tier.T2.zero_recall@10}} |
| T3 explicit content | {{ablation.val.no_t3_vs_full.cold_recall@10}} {{ablation.val.no_t3_vs_full.cold_recall@10.ci}} | {{rq.tier.T3.cold_recall@10}} {{rq.tier.T3.cold_recall@10.ci}} | {{rq.tier.T3.cold_recall@10.reading}} | {{rq.tier.T3.warm_recall@10}} {{rq.tier.T3.warm_recall@10.ci}} | {{rq.tier.T3.warm_recall@10.reading}} | {{rq.tier.T3.zero_recall@10}} |
| U reader match | {{ablation.val.no_u_vs_full.cold_recall@10}} {{ablation.val.no_u_vs_full.cold_recall@10.ci}} | {{rq.tier.U.cold_recall@10}} {{rq.tier.U.cold_recall@10.ci}} | {{rq.tier.U.cold_recall@10.reading}} | {{rq.tier.U.warm_recall@10}} {{rq.tier.U.warm_recall@10.ci}} | {{rq.tier.U.warm_recall@10.reading}} | {{rq.tier.U.zero_recall@10}} |

(The validation column is variant minus full, so a **negative** value means the component helps; the test columns are contributions, full minus variant, so a **positive** value means it helps. Table `rq_part2_tier_contributions`.)

### Step 3: where on the exposure axis (layer 2, finer)

Figure `fig_ablation_heatmap` breaks the change down by how often the article had been shown (0, 1–4, 5–9, 10–19, 20–49, 50+). Red cells mean removing the component lowered recall there, i.e. it helps those articles; a dot marks an interval that excludes zero. Each component helps most at: candidate-aware attention → {{rq.tier.CA.peak_bucket}}; ID masking → {{rq.tier.IDM.peak_bucket}}; T1 → {{rq.tier.T1.peak_bucket}}; T2 → {{rq.tier.T2.peak_bucket}}; T3 → {{rq.tier.T3.peak_bucket}}; U → {{rq.tier.U.peak_bucket}}.

### Step 4: popularity-neutral check (layer 3)

Contribution to within-cold AUC: candidate-aware attention {{rq.tier.CA.cold_auc_within}} ({{rq.tier.CA.cold_auc_within.reading}}), ID masking {{rq.tier.IDM.cold_auc_within}} ({{rq.tier.IDM.cold_auc_within.reading}}), T1 {{rq.tier.T1.cold_auc_within}} ({{rq.tier.T1.cold_auc_within.reading}}), T2 {{rq.tier.T2.cold_auc_within}} ({{rq.tier.T2.cold_auc_within.reading}}), T3 {{rq.tier.T3.cold_auc_within}} ({{rq.tier.T3.cold_auc_within.reading}}), U {{rq.tier.U.cold_auc_within}} ({{rq.tier.U.cold_auc_within.reading}}). A component that helps cold recall *and* within-cold AUC helps the model pick the *right* cold articles; one that helps only recall moves cold articles up as a group.

### Step 5: hypothesis check

| Component | Expected (section 1.4) | Observed (test) | Match? |
|---|---|---|---|
| T1 | Helps warm and overall, not zero-exposure | warm: {{rq.tier.T1.warm_recall@10.reading}}; cold: {{rq.tier.T1.cold_recall@10.reading}}; peak at {{rq.tier.T1.peak_bucket}} | [INTERPRET] |
| T2 | Helps cold (0–9 exposures) | cold: {{rq.tier.T2.cold_recall@10.reading}}; warm: {{rq.tier.T2.warm_recall@10.reading}}; peak at {{rq.tier.T2.peak_bucket}} | [INTERPRET] |
| T3 | Helps zero-exposure and within-cold | zero: {{rq.tier.T3.zero_recall@10}}; within-cold AUC: {{rq.tier.T3.cold_auc_within.reading}} | [INTERPRET] |
| U | Every layer, most for readers with history | overall: {{rq.tier.U.ndcg@10.reading}}; cold: {{rq.tier.U.cold_recall@10.reading}}; warm: {{rq.tier.U.warm_recall@10.reading}} | [INTERPRET] |
| Candidate-aware attention | Helps cold by letting the candidate pick relevant history | cold: {{rq.tier.CA.cold_recall@10.reading}} (attention weights uniform, section 2.6) | [INTERPRET] |
| ID masking | Helps cold by forcing the network off ID look-ups | cold: {{rq.tier.IDM.cold_recall@10.reading}}; overall: {{rq.tier.IDM.ndcg@10.reading}} | [INTERPRET] |

### Step 6: what the ablation cannot tell us

* **Overlap.** Content enters through the news encoder in every variant, and through T3, T2 and U as engineered columns. Removing a tier removes only its engineered columns, so "no detectable effect" for T2 or T3 means "redundant given the encoders", not "content does not matter".
* **One training seed** per variant. Validation Cold Recall@10 of the full model moved by up to 0.12 between epochs, so training randomness on cold metrics may be as large as the bootstrap intervals, which cover only which readers were sampled.
* **Decision rule** (fixed before test): a component "helps cold articles" only if removing it lowers Cold Recall@10 with the validation interval below zero and the test change has the same sign (Table `ablation_decisions`): candidate-aware attention **{{ablation.no_candidate_attention.decision}}**, ID masking **{{ablation.no_id_masking.decision}}**, T1 **{{ablation.no_t1.decision}}**, T2 **{{ablation.no_t2.decision}}**, T3 **{{ablation.no_t3.decision}}**, U **{{ablation.no_u.decision}}**.

# 4. Segment diagnostics: for whom, and when?

Chapter 3 says which component matters. This chapter asks **for which articles, readers and moments** the main model gains or loses.

### Step 1: by article exposure

Figure `fig_recall_by_exposure`, Table `segment_exposure`. For never-shown articles the main model reaches Recall@10 {{test.seg.exposure.0.deep_hybrid.recall@10}} (best simple baseline {{test.seg.exposure.0.bestbase.recall@10}}, NRMS {{test.seg.exposure.0.nrms_g10.recall@10}}). For the most-shown (50+) articles it reaches {{test.seg.exposure.50+.deep_hybrid.recall@10}} ({{test.seg.exposure.50+.bestbase.recall@10}}; NRMS {{test.seg.exposure.50+.nrms_g10.recall@10}}). [INTERPRET] The gap between the curves shows where content takes over from T1.

### Step 2: by reader history

Table `segment_history`. U and the candidate-aware match need a reading history. For readers with no history, Cold Recall@10 is {{test.seg.history.0.deep_hybrid.cold_recall@10}} for the main model and {{test.seg.history.0.bestbase.cold_recall@10}} for the best simple baseline; for readers with 50+ past clicks it is {{test.seg.history.50+.deep_hybrid.cold_recall@10}} and {{test.seg.history.50+.bestbase.cold_recall@10}}. [INTERPRET] If the advantage grows with history, personalisation is doing the work; if it is flat, the article-side evidence is.

### Step 3: by category

Table `segment_category` and `business_category_gain`. {{business.test.n_categories_gain}} of {{business.test.n_categories}} categories gain detectably in Cold Recall@10 over the best simple baseline and {{business.test.n_categories_loss}} lose. The largest gain: {{business.test.top_category}} ({{business.test.top_category.gain.per100}} points, {{business.test.top_category.ci}}); the weakest: {{business.test.weakest_category}} ({{business.test.weakest_category.gain}}). [INTERPRET] Name the well-supported gains with their impression counts.

### Step 4: by slate length and time of day

Tables `segment_slate` and `segment_hour`, reported as lift over random so long and short slates are comparable. [INTERPRET] Long slates are where ranking matters most; time of day tests whether the 6-hour momentum features (T1, T2) track the news cycle.

### Step 5: stress test on rebuilt slates

Every impression is rebuilt as 10 articles, 40% warm and 60% cold, keeping every click (Table `stress_slates`, three random rebuilds). This is a resampled, secondary test. [INTERPRET] The model ordering does / does not hold when cold articles are not a small minority.

# 5. Error analysis: why the model still fails on some cold articles

This chapter looks at individual cases, using validation data so the test day stays a confirmation split.

### Step 1: three kinds of case

| Case | Definition | Count |
|---|---|---|
| Success | Clicked cold article ranked in the top 3 | {{error.success.count}} |
| Failure | Clicked cold article ranked in the bottom quarter of its slate | {{error.failure.count}} |
| False alarm | Cold article ranked in the top 3 but not clicked | {{error.false_alarm.count}} |

### Step 2: which evidence was missing, tier by tier

For each tier we compare its features in failures against successes (Table `error_patterns_by_tier`; a standardised difference of 0.2 or more is notable). Notable differences: T1 {{error.tier.T1.notable}}, T2 {{error.tier.T2.notable}}, T3 {{error.tier.T3.notable}}, U {{error.tier.U.notable}}, slate context {{error.tier.Slate.notable}}. The largest:

1. {{error.pattern1}}
2. {{error.pattern2}}
3. {{error.pattern3}}

[INTERPRET] For example, if failures have low reader-match signals (U), the model misses new articles outside a reader's usual interests. Say whether this matches the ablation in chapter 3.

### Step 3: read the cases

`error_samples.csv` holds 30 cases of each type with titles. [INTERPRET] Describe patterns that hold across many cases (topic, title style, missing abstract), not single anecdotes.

### Step 4: what it means for the model

[INTERPRET] One or two sentences on what evidence the model lacks, and which feature (reader novelty propensity, exposure velocity, slate context) would supply it.

# 6. Answer to the research question

> *Does combining T1, T2 and T3 improve personalised ranking for under-exposed articles, and how does each tier's contribution differ between cold and warm articles?*

**Part 1: does combining help?** Against the best simple baseline (behaviour only) the main model is **{{rq.vs_bestbase.verdict}}**: of every 100 clicked cold articles, {{test.deep_hybrid.cold_recall@10.per100}} were on the first screen, against {{test.bestbase.cold_recall@10.per100}}; for warm articles {{test.deep_hybrid.warm_recall@10.per100}} against {{test.bestbase.warm_recall@10.per100}}. Against Group 10's NRMS (content only) it is **{{rq.vs_nrms_g10.verdict}}** (Table `rq_part1_combination`).

**Part 2: how do the tiers differ?**

* **T1 (own behaviour):** on cold articles **{{rq.tier.T1.cold_recall@10.reading}}**, on warm articles **{{rq.tier.T1.warm_recall@10.reading}}**; matters most at {{rq.tier.T1.peak_bucket}} prior exposures.
* **T2 (borrowed behaviour):** on cold articles **{{rq.tier.T2.cold_recall@10.reading}}**, on warm articles **{{rq.tier.T2.warm_recall@10.reading}}**; matters most at {{rq.tier.T2.peak_bucket}}.
* **T3 (explicit content):** on cold articles **{{rq.tier.T3.cold_recall@10.reading}}**, on warm articles **{{rq.tier.T3.warm_recall@10.reading}}**; matters most at {{rq.tier.T3.peak_bucket}}. Content also reaches the model through the news encoder, which every variant keeps.
* **U (reader match):** on cold articles **{{rq.tier.U.cold_recall@10.reading}}**, on warm articles **{{rq.tier.U.warm_recall@10.reading}}**.
* **Candidate-aware match:** on cold articles **{{rq.tier.CA.cold_recall@10.reading}}**, on warm articles **{{rq.tier.CA.warm_recall@10.reading}}**.

[INTERPRET] One-paragraph answer. Template: "Yes / partly / no. Combining a content-and-history network with behavioural and matching features raises cold-article recall by X over the behaviour-only baseline and by Y over the content-only network, with [no loss / a gain] on warm articles. Own behaviour (T1) carries the warm ranking; cold articles are placed by reading their content and matching it to the reader. The result holds / weakens under popularity-neutral metrics and at thresholds 5 and 20."

# 7. Business insights

## 7.1 What the numbers mean in plain language

| Technical result (test) | Plain-language meaning |
|---|---|
| Cold Recall@10 = {{test.deep_hybrid.cold_recall@10}} | Of every 100 times a reader clicked a **new** article, the ranker had put it on the first screen {{test.deep_hybrid.cold_recall@10.per100}} times |
| Warm Recall@10 = {{test.deep_hybrid.warm_recall@10}} | Same for **established** articles: {{test.deep_hybrid.warm_recall@10.per100}} in 100 |
| Within-cold AUC = {{test.deep_hybrid.cold_auc_within}} | Given two new articles, one clicked and one not, the ranker puts the clicked one higher {{test.deep_hybrid.cold_auc_within.per100}}% of the time (a coin flip is 50%) |
| Least-shown 80% get {{conc.test.deep_hybrid.bottom}} of first-screen slots | Against {{conc.test.random.bottom}} for a random top 10 and {{conc.test.logged.bottom}} of the impressions MSN logged |
| Change in clicked cold articles in the top 10, per 1,000 visits: {{business.test.deep_hybrid_vs_bestbase.n_cold_hits@10}} | Per 1,000 page views, the number of wanted new articles on the first screen changes by {{business.test.deep_hybrid_vs_bestbase.n_cold_hits@10}} compared with the best simple ranker |

## 7.2 How a business could use this

| If the goal is … | Then … | Evidence |
|---|---|---|
| More engagement with fresh content | Use the main model; add a 1–2 slot quota for new articles | Layer 2 and the quota table |
| Fair exposure for new publishers or creators | Track "share of first-screen slots for new articles" and "for the least-shown 80%" as separate KPIs | Layer 4 |
| Protect core engagement | Cap the quota where warm loss passes the agreed tolerance | Layer 5, balance rule |
| Decide where to spend data-engineering effort | Real-time article counters, entity tagging and reading histories; category-level CTR feeds only if T2 helps | Ablation, chapter 3 |
| Readers with no history | Expect weaker personalisation; rely on content signals and the quota | Segment step 2 |

## 7.3 What this does not show

These are offline results on logged data: we re-ordered slates MSN had already chosen and counted where articles readers *did* click would have appeared. They do not show how many extra clicks a live system would get, because readers might click different things when shown a different first screen, and we never see articles MSN chose not to show. The next step for a business would be an online A/B test with a small slot quota, measuring click-through, time spent and the exposure share of new articles.

# 8. Limitations

* Candidates were chosen by MSN's own ranker; we only observe articles that were shown.
* Exposure counters keep updating through test; the test day is not a frozen snapshot.
* Test is one day (14 November 2019).
* Position on the page is not available as a clean feature.
* Prior exposures are counted from the start of the log (9 November), so older articles can look cold on the first day.
* Trained models chose their stopping epoch on validation, so validation numbers favour them; test is the clean comparison.
* One training seed per deep model. Validation Cold Recall@10 of the main model moved between 0.60 and 0.72 across epochs, so cold differences of 2 to 3 points between deep runs may partly be training noise.
* The candidate-aware attention learned uniform weights, so the model's attention is not evidence that it picks out relevant past clicks.

# Appendix A: metric dictionary by layer

| Layer | Metric | Definition | Impressions it is computed on |
|---|---|---|---|
| 1 | AUC | Probability a clicked article is ranked above an unclicked one (ties ½) | At least one click and one non-click |
| 1 | MRR | Mean of 1/rank over clicked articles | At least one click |
| 1 | nDCG@5, @10 | Clicks near the top score more (log discount); 1 = all clicks first | At least one click |
| 2 | Cold / Warm / Zero Recall@k | Share of clicked cold (warm, never-shown) articles in the top k of the full slate | At least one clicked article of that type |
| 2 | Cold Hit@k | 1 if any clicked cold article is in the top k | At least one clicked cold article |
| 3 | Within-cold AUC | AUC using only the slate's cold articles | At least one clicked and one unclicked cold article |
| 3 | Cold-vs-all AUC | Probability a clicked cold article is ranked above any unclicked article | At least one clicked cold article and one non-click |
| 3 | Cold-only nDCG@10 | nDCG after removing warm articles from the slate | As within-cold AUC |
| 3 | Popularity-debiased nDCG@10 | nDCG with click gain 1 / (prior exposures + 1)^γ, γ = 0.5 and 1 | At least one click |
| 4 | Cold slot share@10 | Share of first-screen slots held by cold articles | All |
| 4 | Exposure ratio | Cold slot share / cold share of candidates (1 = proportional) | All |
| 4 | Exposure concentration | Share of first-screen slots for the most-shown 1% and least-shown 80% of articles | All |
| 4 | Coverage@10, Novelty@10 | Distinct articles ever on a first screen / all candidate articles; mean −log2 popularity of the first screen | All |
| 5 | Slot quota *q* | Reserve *q* of the first screen for the best-scoring cold articles | All |
| all | Lift over random | Metric minus a random ranker's expected value on the same impressions | As the base metric |

Full definitions and eligibility rules: `docs/evaluation_spec.md`, section 5.
