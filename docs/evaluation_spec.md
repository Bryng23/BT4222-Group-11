# Evaluation specification v1.0

**Owner:** Shi Ting
**Implements it:** `shared/evaluator.py`, `shared/segments.py`, `shared/audit.py`, `shared/diagnostics.py`, `step6_evaluation_diagnostics.ipynb`
**Verified by:** `tests/test_evaluator.py` (every metric equals a brute-force loop; the shared columns equal `eval_interim`)

Every rule below was fixed on validation before the test day was read.

---

## 1. Data and splits

| Item | Rule |
|---|---|
| Dataset | MIND-small **train** release (9–14 Nov 2019), split by time |
| Splits | train = 9–12 Nov, validation = 13 Nov, test = 14 Nov (`schema.VAL_START`, `schema.TEST_START`) |
| Unit of evaluation | One impression = one slate of candidates shown to one user at one time |
| Candidates | Full logged slates (2–299 candidates). No negative sampling at evaluation time |
| Join | Scores are joined to `meta_{split}.parquet` on `row_id`, never by position. Files that fail `audit.check_predictions` are not scored |
| Which files count | Only frozen choices: prediction files whose metadata say `selected_on = val` |

## 2. Cold definition

| Item | Rule |
|---|---|
| Cold at threshold *t* | Candidate has **fewer than *t* prior exposures** at impression time (`prior_exposures < t`) |
| Prior exposure | An earlier impression in the log (from 9 Nov) in which the article was a candidate. Rows at the same timestamp do not count |
| Primary threshold | *t* = 10 |
| Sensitivity | *t* = 5 and 20. Every cold/warm table is repeated at all three |
| Zero-exposure | `prior_exposures = 0`; reported separately as its own recall (`zero_recall@k`), and as the first exposure bucket |
| Warm | Not cold at the same threshold |
| Threshold justification | Measured on prior exposures (what the flag uses), not on total impressions: (a) error of early CTR as a predictor of later CTR at 5 / 10 / 20 exposures; (b) median hours from first exposure to reaching *t*, and share of articles that ever reach it (`audit.threshold_volume`, `audit.ctr_noise_curve`) |

## 3. Counter policy

Exposure and click counters **keep updating through validation and test**: an impression on 14 Nov sees counts that include 13 Nov and earlier 14 Nov events. This mimics a live system that logs as it serves. It is stated as a limitation: evaluation events update the statistics that later evaluation rows use, so the test day is not a frozen snapshot. Labels of the current impression are never visible to its own features (contract v0.1: strictly earlier events only).

## 4. Ranking and ties

* Rank candidates within each impression by score, descending.
* Ties in the **ranking** are broken by `tie_key(row_id, seed=4222)`, a hash that ignores labels and is identical for every model. A constant-score model therefore ranks in a fixed random order.
* **AUC-type** metrics count tied scores as ½ (Mann–Whitney), as in MIND.

## 5. Metric dictionary

Each metric is computed per impression, NaN where the impression is not eligible, then averaged over eligible impressions. Eligible counts are reported on every table.

The report groups these metrics into six layers, each answering one question: **0 trust** (support, audit), **1 overall** (AUC, MRR, nDCG), **2 cold vs warm** (cold / warm / zero Recall, Cold Hit), **3 popularity-neutral** (within-cold and cold-vs-all AUC, cold-only and popularity-debiased nDCG), **4 exposure** (slot share, exposure ratio, concentration of first-screen slots, coverage, novelty) and **5 balance** (trade-off, slot quota, balance rule). See `docs/report_sections_template.md`, section 1.3.

| Metric | Definition | Eligible impressions | Why it exists |
|---|---|---|---|
| AUC | P(clicked candidate outscores unclicked one), ties ½ | ≥1 click and ≥1 non-click | MIND standard, comparable with the literature |
| MRR | Mean of 1/rank over clicked candidates | ≥1 click | MIND standard |
| nDCG@5, @10 | Binary gain, log2 discount, ideal = all clicks first | ≥1 click | MIND standard; validation selection metric |
| Cold Recall@5, @10 | Clicked cold candidates in the full-slate top *k* / clicked cold candidates | ≥1 clicked cold candidate | Main cold-start result, with warm competitors present |
| Cold Hit@5, @10 | 1 if any clicked cold candidate is in the top *k* | same as Cold Recall | Easier-to-read version of Cold Recall |
| Warm Recall@5, @10 | Same as Cold Recall for warm candidates | ≥1 clicked warm candidate | The warm side of the trade-off |
| Zero Recall@5, @10 | Same for zero-exposure candidates | ≥1 clicked zero-exposure candidate | Strictest cold case |
| Cold AUC, within-cold | AUC using only the impression's cold candidates | ≥1 clicked and ≥1 unclicked cold candidate | Popularity-neutral: no warm item to lean on |
| Cold AUC, cold-vs-all | P(clicked cold candidate outscores any unclicked candidate) | ≥1 clicked cold candidate and ≥1 non-click | Can a cold item be surfaced above warm ones? |
| Cold-only nDCG@10 | nDCG after dropping warm candidates (cold keep their order) | same as within-cold AUC | The "separate nDCG for cold" the TAs asked for |
| Popularity-debiased nDCG@10, γ | Click gain = 1 / (prior exposures + 1)^γ; normalised by the same impression's ideal ordering of those gains. γ = 0.5 and 1 (γ = 1 is quoted in the report) | ≥1 click | nDCG that does not reward popular items |
| Cold slot share@10 | Cold candidates in top 10 / min(10, slate size) | all | Rich-get-richer effect, read against cold candidate share |
| Cold exposure ratio@10 | mean cold slot share / mean cold candidate share | all | 1 = cold items get slots in proportion to their presence |
| Novelty@10 | Mean over top 10 of −log2 *p*, *p* = share of the split's impressions in which the article is a candidate | all | Business-facing diversity |
| Coverage@10 | Distinct articles in any top 10 / distinct candidate articles in the split | split-level | Business-facing diversity |
| Lift over random | Metric minus a random ranker's expected value on the same eligible impressions. Recall@k: min(k, n)/n per impression; AUC: 0.5 | same as base metric | Slate lengths vary 2–299, so raw recall is not comparable across segments |

Segment metrics (`segments.py`): for candidate-level segments (exposure bucket, category), Recall@k / Hit@k / AUC-vs-all of clicked candidates **in that segment**, in full slates. Exposure buckets: 0, 1–4, 5–9, 10–19, 20–49, 50+. Impression-level segments (user history 0 / 1–4 / 5–19 / 20–49 / 50+; slate 2–9 / 10–24 / 25–49 / 50–99 / 100+; hour of day in 6-hour blocks): every metric within the segment.

## 6. Uncertainty

| Item | Rule |
|---|---|
| Interval | 95% percentile bootstrap, 1,000 resamples, seed 0 |
| Resampling unit | **User** (`user_id`). All impressions of a resampled user move together. Impressions of one user are correlated, so resampling impressions gives intervals that are too narrow |
| Paired comparisons | Model B − model A on the same impressions; the statistic is recomputed on each resample of users |
| Training noise | Not covered by the bootstrap. Reported as a seed spread when step 5 writes seeded files; otherwise every model is stated as single-seed |
| Significance language | "Better / worse" only when the interval excludes 0; otherwise "no detectable difference". No p-values |

## 7. Selection and balance rule

* **Current rule (all tuned models, `selection.py`)**: highest validation Cold Recall@10 among settings within 0.005 validation nDCG@10 of the best. The 0.005 is provisional.
* **Balance rule**: maximise validation Cold Recall@10 subject to warm Recall@10 ≥ (1 − *x*) × the best baseline's validation warm Recall@10. Tolerances *x*: 1%, 2%, 5%. Step 6 reports the choice under each *x* and the Pareto front of all models and slot-quota settings.
* **Slot-quota re-ranking** (no retraining): reserve *q* = 0, 10, 20, 30% of the top 10 for the best-scoring cold candidates; inside the top 10, order by score. Evaluated with rank metrics only (re-ranked scores have no ties, so AUC moves slightly even at *q* = 0).
* The balance rule and *x* are fixed on validation **before** test is read.

## 8. Ablation protocol

* Variants are the step 5 ablations of the main deep model (listed in step 6, section 0); every variant is retrained from scratch with the same seed, inputs, training loop and best-validation-epoch rule as step 4.
* Leave-one-out (`no_*`, compared with `full`) removes one component: T1, T2, T3, U, candidate-aware attention or ID masking. `full_plus_smoothing` adds content smoothed CTR. Components overlap, so contributions do not add up.
* **Decision rule (fixed before test):** a tier *helps cold articles* only if removing it lowers Cold Recall@10 with the validation interval entirely below 0 **and** the test delta has the same sign. The mirror image is *hurts cold*. Anything else is "no detectable effect".
* `no_t3` removes explicit content columns only; content still enters through the news encoder, T2 and U.

## 9. Test protocol

1. Validation drives every decision: hyperparameters, α/β, epochs, balance rule and *x*, γ, which model is "main".
2. Test is read once for confirmation. Nothing is re-tuned afterwards, and any fix must not depend on test results.
3. Error analysis runs on validation, so test stays a confirmation split.

## 10. Reporting

* Every table has a support column (eligible impressions, clicks, cold-click impressions).
* Every result is a delta against `category_ctr`, `candidate_smoothed_ctr` and `nrms_g10` with a user-clustered interval.
* Cold and warm are reported side by side, never as one blended number.
* Train numbers are shown for completeness; for trained models they are in-sample.
* Stress-slate results are labelled **resampled** and are secondary.
* Business numbers are offline counts on logged slates, not predicted online clicks.
