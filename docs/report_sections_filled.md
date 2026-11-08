<!--
Report sections: step 6 run on Colab (deep hybrid main model), with every [INTERPRET] resolved from report_numbers
and the step 6 tables (step6_outputs/real/tables).
-->

# 0. Summary of results and final business insights

> **Read this first.** Numbers are from the **test** day (14 November 2019; 837 impressions with a clicked cold article), scored once after every choice was frozen on validation; validation values are given in brackets where they matter. "Cold" means fewer than 10 prior exposures. Intervals are 95%, paired (same impressions, two models) and resample readers. A difference counts only when its interval excludes zero. "Per 1,000 visits" means per 1,000 impressions.

## 0.1 What each step produced

| Step | What it did | Outcome |
|---|---|---|
| 1–2 Data and features | MIND-small train log (156,965 impressions) split by time into train 95,071 impressions (3.39M candidate rows), validation 31,624 (1.23M) and test 30,270 (1.22M). Features built only from events strictly before each impression | Engineered features in four tiers: T1 own behaviour, T2 borrowed behaviour, T3 pure content, U reader match. 1.9% of test clicks are on cold articles |
| 3a Simple baselines | Random, popularity, category CTR, candidate smoothed CTR, content smoothed CTR; smoothing strengths tuned on validation (β = 5, α = 20) | Best simple baseline (validation nDCG@10): candidate_smoothed_ctr, test nDCG@10 0.3569 |
| 3b Group 10 NRMS | Corrected reimplementation of Group 10's attention network (content and history only); best epoch 10 | Test nDCG@10 0.3439, Cold Recall@10 0.5743 |
| **4 Deep hybrid (main model)** | Group 10's attention network plus an engineered-feature tower (T1/T2/T3/U), candidate-aware attention over the reader's history, a GMF term and an article ID embedding with ID masking (p = 0.3). Best validation epoch 3; trained in 19 minutes on CPU | **Test nDCG@10 0.4279, Cold Recall@10 0.6268** |
| 5 Ablation | Seven variants of the main model, each retrained from scratch with one component removed or added | Section 0.3 |
| 6 Evaluation | Six layers, ablation, segments, error analysis, business translation; nothing trained | This report |

## 0.2 Results by layer

**Headline table (test, top 10 = first screen):**

| Model | AUC | nDCG@10 | Cold Recall@10 | Warm Recall@10 | Never-shown Recall@10 | Within-cold AUC |
|---|---|---|---|---|---|---|
| **Deep hybrid (main)** | **0.7095** | **0.4279** | **0.6268** | **0.6660** | **0.7063** | **0.6475** |
| Group 10 NRMS | 0.6094 | 0.3439 | 0.5743 | 0.5788 | 0.6352 | 0.5920 |
| Best simple baseline (candidate_smoothed_ctr) | 0.6468 | 0.3569 | 0.2489 | 0.5953 | 0.0769 | 0.5496 |
| Content smoothed CTR | 0.6479 | 0.3563 | 0.2694 | 0.5952 | 0.1643 | 0.5748 |
| Category CTR | 0.4977 | 0.2549 | 0.2979 | 0.4808 | 0.2692 | 0.4985 |
| Popularity | 0.4608 | 0.2405 | 0.0287 | 0.4666 | 0.0210 | 0.4944 |
| Random | 0.5000 | 0.2623 | 0.2282 | 0.4844 | 0.2098 | 0.5000 |
| *Validation:* deep hybrid | 0.7163 | 0.4509 | 0.6875 | 0.6856 | 0.7455 | 0.6689 |

**Results by layer.** "References" are the best simple baseline and Group 10's NRMS. "Difference" is main model minus reference, with its 95% interval.

| Layer | Question it answers | Metric (test) | Main model | References | Difference [95% interval] | Verdict | In plain English |
|---|---|---|---|---|---|---|---|
| 0 Trust | Are there enough cold clicks to measure? | Impressions with a clicked cold article | 837 (912 cold clicks) | 506 at threshold 5; 1,276 at threshold 20 | Margin on Cold Recall ≈ ±0.027 | Enough for medium effects | New-article clicks are rare (1.9% of all clicks), but there are over 800 of them. That is enough to see a difference of about 3 in 100, not 1 in 100 |
| 0 Trust | Are the cold labels right? | Agreement with an independent recount of prior exposures | 100% of rows (Table `audit_cold_flags`) | n/a | 0 rows differ | Labels trusted | We counted each article's past views a second way and got the same answer for every row, so "new" means what we say it means |
| 1 Overall | Is it a good ranker in general? | nDCG@10 | 0.4279 | 0.3569; NRMS 0.3439 | +0.0710 [+0.0677, +0.0744]; vs NRMS +0.0840 [+0.0805, +0.0873] | Better than both | Whatever readers clicked, our ranker put it nearer the top of the list than the simple ranker or Group 10's network |
| 1 Overall | Same, ignoring position | AUC | 0.7095 | 0.6468; NRMS 0.6094 | +0.0627 [+0.0593, +0.0660]; vs NRMS +0.1001 [+0.0965, +0.1039] | Better than both | Pick a clicked and an unclicked article from the same page: we rank the clicked one higher 71 times in 100, against 65 for the simple ranker and 61 for NRMS (50 = coin flip) |
| 2 Cold vs warm | Are clicked **new** articles on the first screen? | Cold Recall@10 | 0.6268 | 0.2489; NRMS 0.5743 | +0.3779 [+0.3435, +0.4156]; vs NRMS +0.0526 [+0.0187, +0.0886] | Better than both, for cold and for warm articles | Of every 100 new articles readers clicked, 63 were on our first screen, against 25 for the simple ranker and 57 for NRMS |
| 2 Cold vs warm | ...without losing **established** ones? | Warm Recall@10 | 0.6660 | 0.5953; NRMS 0.5788 | +0.0708 [+0.0654, +0.0757]; vs NRMS +0.0872 [+0.0825, +0.0922] | Better than both | Helping new articles did not cost the established ones: 67 in 100 clicked established articles were on the first screen, against 60 and 58 |
| 2 Cold vs warm | And articles **never shown before**? | Never-shown Recall@10 | 0.7063 | 0.0769; NRMS 0.6352; random 0.2098 | +0.6294 [+0.5513, +0.7080]; vs NRMS +0.0711 [−0.0068, +0.1521] | Better than baseline; no detectable difference from NRMS | For brand-new articles with no views at all: 71 in 100 clicked ones reached the first screen, against 8 for the simple ranker, which is worse than random (21). Only 143 such impressions, so treat as indicative |
| 3 Popularity-neutral | Does it pick the **right** new articles, or push all of them up? | Within-cold AUC | 0.6475 | 0.5496; NRMS 0.5920 | +0.0979 [+0.0621, +0.1355]; vs NRMS +0.0555 [+0.0200, +0.0926] | Better than both | Given two new articles, one clicked and one not, we rank the clicked one higher 65 times in 100 (simple ranker 55, NRMS 59, coin flip 50). The cold gain is real judgement, not a blanket boost |
| 3 Popularity-neutral | Does it still win when popular clicks count less? | Popularity-debiased nDCG@10 (γ = 1) | 0.4232 | 0.3435; NRMS 0.3384 | Table `metric_robustness` | Higher than both | Even when a click on a hugely popular article counts less than a click on an obscure one, our ranker still comes out well ahead, so it is not just riding popularity |
| 4 Exposure | Do new articles get a fair share of the first screen? | Cold exposure ratio (slot share ÷ candidate share; 1 = proportional) | 1.88 | 0.88; popularity 0.05; random 1.00 | Descriptive (no interval) | Above fair share | New articles get 88% more first-screen room than their share of the candidate list. The simple ranker gives them 12% less than their share; popularity gives them almost none |
| 4 Exposure | Does the long tail of articles get seen? | Share of first-screen slots for the least-shown 80% of articles | 3.9% | 2.2%; NRMS 3.5%; random top 10 2.3%; popularity 0.3% | Descriptive | Highest of all rankers | The least-shown 80% of articles get more of our first screen than under any other ranker, including random, and more than their 3.3% share of what MSN logged |
| 5 Balance | What do extra slots for new articles cost? | Cold / Warm Recall@10 with 1 slot of 10 reserved (validation) | 0.7642 / 0.6835 | No quota: 0.6875 / 0.6856 | Per 1,000 test visits: +2.6 new, −4.5 established | Cheap | Reserving one slot raises new articles found from 69 to 76 in 100, while established ones barely move (69 to 68) |
| 5 Balance | Same with 3 slots, and what the rule picks | Cold / Warm Recall@10 with 3 slots of 10 reserved (validation) | 0.8746 / 0.6795 | No quota: 0.6875 / 0.6856 | Per 1,000 test visits: +6.4 new, −13.1 established | Rule picks 3 slots at every tolerance (1%, 2%, 5%) | With three slots, 87 in 100 new articles are found, and established articles stay at 68 in 100. The rule never says stop, so the choice is a business one |

Threshold check (Table `threshold_survival`): against the best baseline and category CTR, the cold and warm gains are detectable at thresholds 5, 10 and 20 on both splits. Against NRMS the warm gain holds everywhere and the cold gain points the same way at every threshold. It is detectable on validation at all three and on test at 10 and 20, but not at 5 on test (+0.028 [−0.020, +0.076], 506 impressions).

## 0.3 Ablation: which part of the model is responsible?

Contribution = main model minus the variant without that component (positive = the component helps), on test. "Helps" only when the interval excludes zero. Every variant keeps the attention news and user encoders.

| Component removed | nDCG@10 | Cold Recall@10 | Reading (cold) | Warm Recall@10 | Reading (warm) | Never-shown Recall@10 | Within-cold AUC | Matters most at (prior exposures) | Decision rule (cold) |
|---|---|---|---|---|---|---|---|---|---|
| Candidate-aware attention | +0.0030 [+0.0019, +0.0041] | +0.0282 [+0.0137, +0.0446] | helps | +0.0047 [+0.0027, +0.0065] | helps | +0.0548 | +0.0145 (helps) | 0 | helps cold |
| ID masking | +0.0039 [+0.0025, +0.0054] | +0.0046 [−0.0139, +0.0248] | no detectable effect | +0.0069 [+0.0044, +0.0093] | helps | −0.0350 | +0.0110 | 10–19 | no detectable effect |
| T1 own behaviour | +0.0413 [+0.0389, +0.0438] | −0.0242 [−0.0509, +0.0018] | no detectable effect | +0.0392 [+0.0356, +0.0427] | helps | +0.0408 | +0.0008 | 20–49 | no detectable effect |
| T2 borrowed behaviour | +0.0002 [−0.0009, +0.0014] | +0.0073 [−0.0102, +0.0248] | no detectable effect | +0.0012 [−0.0009, +0.0033] | no detectable effect | −0.0210 | +0.0019 | 1–4 | no detectable effect |
| T3 explicit content columns | +0.0043 [+0.0028, +0.0058] | +0.0125 [−0.0060, +0.0313] | no detectable effect | +0.0050 [+0.0024, +0.0076] | helps | +0.0268 | +0.0145 (helps) | 10–19 | no detectable effect |
| U reader match | +0.0252 [+0.0222, +0.0280] | +0.0113 [−0.0193, +0.0473] | no detectable effect (validation: helps) | +0.0270 [+0.0232, +0.0312] | helps | +0.0594 | +0.0354 (helps) | 10–19 | helps cold |

Adding content smoothed CTR as a feature (`full_plus_smoothing`) changes test Cold Recall@10 by +0.0107 [−0.0078, +0.0295] and nDCG@10 by +0.0003 [−0.0008, +0.0014]: no detectable effect.

What the ablation shows:

* **New articles are found mainly by the network that reads them and the reader, not by any single engineered tier.** Removing T1, T2, T3 or U leaves test Cold Recall@10 unchanged within its interval. The content and history encoders are in every variant; Group 10's content-only network alone already reaches 0.5743 test Cold Recall@10.
* **Candidate-aware attention is the one component that clearly helps cold articles on both splits** (+0.028 on test and on validation), and it matters most for never-shown articles (peak at 0 exposures; +0.055 never-shown recall). Its attention weights stayed uniform (section 2.6), so what it adds is a second, candidate-specific match between the reader's average history and the article, not the selection of particular past clicks.
* **The engineered features help pick the *right* new articles even where they do not find more of them.** U (+0.035), T3 and candidate-aware attention (+0.015 each) all raise within-cold AUC detectably. They sharpen the order among cold articles without changing how many reach the first screen.
* **T1 carries the established-article ranking.** Removing it gives the largest drops in nDCG@10 (−0.041) and Warm Recall@10 (−0.039). It matters most at 20–49 prior exposures and has no detectable cold effect, as section 1.4 expects.
* **U (explicit reader–article match) is the second pillar overall** (nDCG@10 +0.025, Warm Recall@10 +0.027). Its cold contribution is large on validation (+0.072) and points the same way on test without reaching significance (+0.011), which is what the decision rule counts as "helps cold".
* **ID masking works as a regulariser, not as a cold-start device.** It improves overall ranking and warm recall but not cold recall. Without masking, the network peaked at epoch 1 and then overfitted.
* **T2 and the smoothed-CTR feature add nothing detectable** once the rest of the model is present.

## 0.4 Segments and errors

* **By article exposure:** for never-shown articles the main model reaches Recall@10 0.7063, against 0.0769 for the best simple baseline and 0.6352 for NRMS. For the most-shown (50+) articles it is 0.6636, against 0.5963 and 0.5766. The gap over the baseline is largest for new articles (+0.63) and smallest for heavily-shown ones (+0.07) (Figure `fig_recall_by_exposure`).
* **By reader history:** the cold advantage over the best baseline is +0.31 to +0.45 in every history bucket with real support, and does not grow with history (50+ past clicks: 0.6022 vs 0.2383). The gain comes mainly from the article side (reading its content), not from personalisation alone. For the heaviest readers (50+), Group 10's content-only network is level with the main model on new articles (0.6276 vs 0.6022). Readers with no history show 0.7778 vs 0.3889, but on only 18 cold-click impressions.
* **By category:** 11 of the 14 categories with clicked cold articles gain detectably in Cold Recall@10 over the best baseline, and none lose. The well-supported gains are sports (+47 points, 261 impressions) and news (+36 points, 310 impressions); finance gains +45 points (40 impressions). The largest point gain, video (+54 points), rests on 24 impressions.
* **By slate length and time of day:** in every slate of 10 or more articles, the main model's cold recall is 0.32 to 0.48 above random, while the best baseline's is about zero (−0.10 to +0.09). Its cold recall is steady across the day (0.61 to 0.66); NRMS drops to 0.48 overnight.
* **Stress slates** (every slate rebuilt as 10 articles, 60% cold; resampled): the main model finds the most new articles (cold Recall@5 0.75 vs 0.71 for NRMS and 0.49 for the baseline) and orders them best (within-cold AUC 0.65). On overall nDCG it falls behind popularity (0.62 vs 0.72), because these slates keep every click, and clicks there are mostly on the warm articles that popularity puts first.
* **Errors (validation):** 356 clicked cold articles were ranked in the top 3 and only 9 in the bottom quarter. In those 9 failures, the reader's match with the article is far weaker: text similarity to the reader's history 0.108 vs 0.359 in successes, and almost no affinity for the article's category (0.025 vs 0.391). The failures are mostly lifestyle, food, TV and finance pieces shown to readers who never read those categories. The model misses new articles that fall outside a reader's usual interests.

## 0.5 Answer to the research question (test)

**Part 1: yes, combining content, behaviour and reader evidence helps new articles.** Against the best simple baseline the main model is **better for cold articles and better for warm articles**. Of every 100 clicked new articles, it put 63 on the first screen, against 25 for the best simple baseline and 57 for Group 10's content-only network. For established articles: 67 vs 60 and 58. Against NRMS it is also **better for cold articles and better for warm articles**. It also picks the right new articles more often (within-cold AUC 0.65 vs 0.55 and 0.59).

**Part 2: the evidence does different jobs.** T1 (own behaviour) carries the warm ranking (**helps** warm, **no detectable effect** on cold). U (reader match) **helps** warm articles and, under the decision rule, cold ones (detectable on validation, same direction on test). The candidate-aware match **helps** cold articles on both splits. T2 and T3 have **no detectable effect** on cold recall, although T3 and U help the model order new articles correctly (within-cold AUC).

## 0.6 Final business insights

These translate the test results for a news editor or product manager. All are offline estimates: we re-ordered slates MSN had already chosen and counted where the articles readers did click would have appeared.

**1. New articles can be surfaced before they have a click history, without hurting established ones.** Per 1,000 visits, the main model puts **+11.4 more wanted new articles** on the first screen than the best simple ranker [+10.1, +12.7], which more than doubles the count, and **+104.5 more wanted established articles** [+97.2, +111.8]. Ranking by popularity, the common default, finds only 3 in 100 wanted new articles, worse than random (23 in 100).

**2. Reading the article and the reader is what makes new articles rankable; the article's own counters keep established ones ranked well.** The content-only network (NRMS) finds 57 of 100 new articles but only 58 established ones. The behaviour-only baseline is the reverse (25 and 60). The main model combines both: 63 new and 67 established. Against NRMS that is +1.7 wanted new articles [+0.6, +2.7] and +140.4 established ones [+133.0, +147.9] per 1,000 visits.

**3. Invest in article metadata, reading histories and real-time article counters; borrowed category click feeds are optional.** Removing borrowed category and topic click rates (T2) has no detectable effect on cold articles or overall. The article's own counters (T1) and the reader–article match (U) are the largest contributors overall (nDCG@10 +0.0413 and +0.0252). For a newsroom, titles, abstracts, entity tags, clean reading histories and live click counters matter more than streaming category-level CTR pipelines.

**4. There is a dial, and its price is known.** Reserving first-screen slots for the best-scoring new articles, per 1,000 test visits:

| Slots reserved | Extra wanted new articles | Change in wanted established articles | New gained per established lost |
|---|---|---|---|
| 1 of 10 | +2.6 | −4.5 | 0.58 |
| 2 of 10 | +4.9 | −9.2 | 0.53 |
| 3 of 10 | +6.4 | −13.1 | 0.49 |

Each slot buys a little less than the one before. Against about 891 wanted established articles shown per 1,000 visits, even 3 slots costs under 1.5% of them. The balance rule (warm Recall@10 within 1, 2 or 5% of the best baseline's) picks 3 slots at every tolerance, because the main model's warm recall is far above the baseline's: the rule does not bind, so the choice is a business one. Because the model already gives new articles 1.9 times their fair share without a quota, we recommend **at most 1–2 slots** as a starting point.

**5. The model loosens the rich-get-richer loop.** It gives new articles 1.88 times their share of candidates on the first screen. It also gives the least-shown 80% of articles 3.9% of first-screen slots, more than any other ranker (random top 10: 2.3%; simple ranker: 2.2%; popularity: 0.3%) and more than their 3.3% share of MSN's logged impressions. The most-shown 1% get 49.9% of its first screen, less than under a random top 10 (52.2%). A business that cares about creator or publisher fairness should still track both KPIs separately: the share of first-screen slots for new articles, and for the least-shown 80%.

**What to do next.** Run an online A/B test of the main model, with and without a 1-slot quota for new articles, measuring click-through, time spent, cold slot share and long-tail share against the current ranker.

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
  * *Noise.* Early CTR is a poor guide to an article's later CTR. Its error falls to 0.79 of the one-exposure value after 5 exposures, 0.65 after 10 and 0.49 after 20.
  * *Time.* The median article reaches 10 exposures 3.9 hours after it is first shown (5 exposures: 2.3 h; 20: 5.3 h). Only 43.5% of articles ever reach 10.
  * So "cold" covers roughly an article's first 3.9 hours: the window in which its own clicks are still too noisy to rank on.
* **Counters** keep updating through validation and test, as in a live system.
* **Ties** are broken by a label-blind hash, identical for every model.
* **Uncertainty.** Every comparison is paired (same impressions, two models) with a 95% interval that resamples *readers*, because one reader's impressions are related. We say "better" or "worse" only when the interval excludes zero; otherwise "no detectable difference". 1,000 resamples.
* **Support.** Every table shows how many impressions and clicks each number rests on.
* **References.** Category CTR, candidate smoothed CTR, Group 10's NRMS, and the strongest simple baseline on validation (candidate_smoothed_ctr).

# 2. Results, layer by layer

All numbers are on the test day unless marked. Full train / validation / test tables: `headline_all_models`; all differences: `headline_deltas`, Figure `fig_forest_main_vs_references`.

## Layer 0: can we trust the test?

The test day has **837** impressions with at least one clicked cold article (912 cold clicks; validation 938). With that many, a Cold Recall difference has a margin of error of roughly ±0.027. At the stricter threshold of 5 the count falls to 506; at 20 it rises to 1,276 (Table `support_cold_clicks`). The cold flags were recounted independently from the raw log and agree with the stored flags on every row (Table `audit_cold_flags`). The test day is therefore enough to read medium-sized cold effects (about 3 points of recall), not small ones. Never-shown articles (143 test impressions) are thinner and read as indicative only.

## Layer 1: overall ranking

| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| **Deep hybrid (main)** | 0.7095 | 0.3366 | 0.3672 | 0.4279 |
| Group 10 NRMS | 0.6094 | 0.2658 | 0.2815 | 0.3439 |
| Best simple baseline (candidate_smoothed_ctr) | 0.6468 | 0.2784 | 0.2937 | 0.3569 |
| Content smoothed CTR | 0.6479 | 0.2775 | 0.2929 | 0.3563 |
| Category CTR | 0.4977 | 0.1945 | 0.1929 | 0.2549 |
| Popularity | 0.4608 | 0.1822 | 0.1792 | 0.2405 |
| Random | 0.5000 | 0.2026 | 0.2034 | 0.2623 |

Against the best simple baseline the main model changes nDCG@10 by +0.0710 [+0.0677, +0.0744]; against NRMS by +0.0840 [+0.0805, +0.0873] (validation +0.0680 [+0.0647, +0.0713]).

**Training vs test.** Train nDCG@10 is 0.5141, validation 0.4509, test 0.4279 (NRMS: 0.4988, 0.3829, 0.3439). Training-split values are in-sample. The main model's drop from train to test (0.086) is about half of NRMS's (0.155). Early stopping picked epoch 3 of 6; validation nDCG@10 fell after that while training loss kept dropping.

*Reading:* this layer is about 98% warm clicks, so it shows the model is a competent general ranker, clearly better than both references, but says nothing yet about cold articles.

## Layer 2: cold vs warm vs never-shown

| Model | Cold Recall@10 | Cold Hit@10 | Warm Recall@10 | Zero-exposure Recall@10 |
|---|---|---|---|---|
| **Deep hybrid (main)** | **0.6268** | 0.6464 | 0.6660 | 0.7063 |
| Group 10 NRMS | 0.5743 | 0.5926 | 0.5788 | 0.6352 |
| Best simple baseline | 0.2489 | 0.2593 | 0.5953 | 0.0769 |
| Content smoothed CTR | 0.2694 | 0.2784 | 0.5952 | 0.1643 |
| Category CTR | 0.2979 | 0.3059 | 0.4808 | 0.2692 |
| Popularity | 0.0287 | 0.0287 | 0.4666 | 0.0210 |
| Random | 0.2282 | 0.2378 | 0.4844 | 0.2098 |

* **vs the best simple baseline:** cold +0.3779 [+0.3435, +0.4156]; warm +0.0708 [+0.0654, +0.0757]. **Better for cold articles; better for warm articles.**
* **vs Group 10's NRMS:** cold +0.0526 [+0.0187, +0.0886] (validation +0.1071 [+0.0748, +0.1398]); warm +0.0872 [+0.0825, +0.0922]. **Better for cold articles; better for warm articles.** The engineered tower did not cost the content network its cold strength; it added to it.
* **vs category CTR (the baseline best at finding new articles):** cold +0.3290 [+0.2935, +0.3629]. **Better for cold articles; better for warm articles.**
* **Never-shown articles:** 0.7063, against 0.0769 for the best baseline (+0.63 [+0.55, +0.71]) and 0.6352 for NRMS (+0.07 [−0.01, +0.15], not detectable). The smoothed-CTR baselines are *worse than random* here, because a never-shown article has no clicks to smooth.

*Threshold check* (Tables `threshold_robustness_deltas`, `threshold_survival`), main model minus reference, test:

| Threshold | Cold-click impressions | vs best baseline, cold | vs NRMS, cold | vs NRMS, warm |
|---|---|---|---|---|
| 5 | 506 | +0.4858 [+0.4405, +0.5333] | +0.0280 [−0.0199, +0.0763] | +0.0873 [+0.0826, +0.0923] |
| 10 | 837 | +0.3779 [+0.3435, +0.4156] | +0.0526 [+0.0187, +0.0886] | +0.0872 [+0.0825, +0.0922] |
| 20 | 1,276 | +0.3183 [+0.2880, +0.3498] | +0.0673 [+0.0389, +0.0983] | +0.0874 [+0.0825, +0.0923] |

Against the best baseline and category CTR, every cold and warm conclusion survives at all three thresholds on both splits. Against NRMS the warm gain survives everywhere. The cold gain has the same sign at every threshold and is detectable on validation at all three (+0.12, +0.11, +0.12). On test it is not detectable at the strictest threshold, 5. There, the content-only network is closest, because articles with fewer than 5 exposures have almost no behaviour for the engineered features to use.

*A note on variance.* Validation Cold Recall@10 of the main model swung between 0.60 and 0.72 from epoch to epoch (step 4). Cold differences of 2 to 3 points between deep runs are within training noise.

## Layer 3: is the cold result real?

| Model | Within-cold AUC | Cold-vs-all AUC | Cold-only nDCG@10 | Popularity-debiased nDCG@10 (γ = 1) |
|---|---|---|---|---|
| **Deep hybrid (main)** | 0.6475 | 0.8248 | 0.7067 | 0.4232 |
| Group 10 NRMS | 0.5920 | 0.7511 | 0.6715 | 0.3384 |
| Best simple baseline | 0.5496 | 0.5244 | 0.6366 | 0.3435 |
| Content smoothed CTR | 0.5748 | 0.6124 | 0.6546 | 0.3412 |
| Category CTR | 0.4985 | 0.5938 | 0.6153 | 0.2496 |
| Random | 0.5000 | 0.5000 | 0.6157 | 0.2557 |

Within-cold AUC difference: vs best baseline +0.0979 [+0.0621, +0.1355]; vs NRMS +0.0555 [+0.0200, +0.0926]. Table `metric_robustness` repeats every comparison under each popularity-neutral metric.

Layers 2 and 3 agree. The main model finds more new articles *and* orders them better: given a clicked and an unclicked new article in the same slate, it ranks the clicked one higher 65 times in 100, against 55 for the best baseline and 59 for NRMS. The cold gain comes from telling good new articles from bad ones, not from lifting all of them. A clicked new article also beats unclicked articles of any kind 82 times in 100 (cold-vs-all AUC 0.82), so good new articles can compete with popular ones on merit. The popularity-debiased nDCG puts the main model just as clearly ahead (0.42 vs 0.34), so the overall result is not a reward for ranking popular articles first.

## Layer 4: who gets the first screen?

| Share of exposure going to … | Logged (what MSN showed) | Deep hybrid, top 10 | NRMS, top 10 | Best simple baseline, top 10 | Popularity, top 10 | Random, top 10 |
|---|---|---|---|---|---|---|
| Most-shown 1% of articles | 36.8% | 49.9% | 53.8% | 67.0% | 77.3% | 52.2% |
| Least-shown 80% of articles | 3.3% | 3.9% | 3.5% | 2.2% | 0.3% | 2.3% |

Cold articles hold 2.35% of the main model's first-screen slots. The exposure ratio (their slot share divided by their share of candidates; 1 = proportional) is 1.88, against 0.88 for the best simple baseline, 0.05 for the popularity ranker and 1.00 for random. Coverage@10: 0.636 of all candidate articles reach a first screen at least once (random 0.590). Table `exposure_concentration`.

*Reading.* Any top-10 ranker concentrates exposure more than the logged slates, because the hits appear in many slates at once, so the like-for-like reference is the random top 10. Against it, the main model gives the most-shown 1% *less* of the first screen (49.9% vs 52.2%) and the least-shown 80% *more* (3.9% vs 2.3%). Its long-tail share is the highest of any ranker and above what MSN logged. The best simple ranker does the opposite (67.0% to the top 1%). The main model loosens the rich-get-richer loop rather than feeding it.

## Layer 5: balance

Figure `fig_tradeoff_val` places every model by warm Recall@10 (x) and cold Recall@10 (y); up and to the right is better on both. The lines show what happens when part of the first screen is reserved for the best-scoring cold articles (no retraining):

| First-screen slots reserved for cold | Cold Recall@10 | Warm Recall@10 |
|---|---|---|
| 0% (model as trained) | 0.6875 | 0.6856 |
| 10% (1 of 10) | 0.7642 | 0.6835 |
| 20% (2 of 10) | 0.8326 | 0.6811 |
| 30% (3 of 10) | 0.8746 | 0.6795 |

(Validation.) Under the balance rule "highest cold Recall@10 while warm Recall@10 stays within 2% of the best baseline (candidate_smoothed_ctr, 0.6166)", the chosen setting is **3 slots (q = 30%)**, at 1% and 5% as well (Table `balance_rule`). The rule does not bind: even with 3 slots the main model's warm recall (0.680) stays 10% above the baseline's. The curve is steep: the first slot buys 7.7 points of cold recall for 0.2 points of warm recall.

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
| 0 Trust | 837 cold-click impressions and verified cold labels: enough for medium effects |
| 1 Overall | nDCG@10 0.4279, better than the best baseline (0.3569) and NRMS (0.3439) |
| 2 Cold vs warm | Better for cold and for warm articles, against both the best simple baseline and NRMS |
| 3 Popularity-neutral | Within-cold AUC 0.6475 vs 0.5496 (baseline) and 0.5920 (NRMS): agrees with layer 2, so the cold gain is genuine judgement |
| 4 Exposure | Least-shown 80% get 3.9% of first-screen slots, the most of any ranker (random top 10: 2.3%; logged: 3.3%) |
| 5 Balance | Rule picks 3 slots at every tolerance (does not bind); first slot is cheap |

# 3. Ablation: which evidence is responsible?

Chapter 2 says *what* the main model does. This chapter asks *which component causes it*. Each variant is the main model retrained from scratch with one change, using the same seed, inputs, training loop and best-validation-epoch rule (step 5; Table `ablation_variants`).

| Variant | What changes | Best epoch (step 5) |
|---|---|---|
| `full` | Main model (step 4) | 3 |
| `no_candidate_attention` | Without the candidate-aware attention branch | run in an earlier step 5 session |
| `no_id_masking` | ID vectors never masked in training | 1 |
| `no_t1` | Without the article's own behaviour | 3 |
| `no_t2` | Without borrowed behaviour | 3 |
| `no_t3` | Without explicit content columns (codes, entity counts) | 3 |
| `no_u` | Without the engineered reader–article matching columns | 5 |
| `full_plus_smoothing` | Adds content smoothed CTR (α = 20) | 3 |

**Contribution** of a component = full model minus the model without it. **Positive means it helps.** A contribution counts only if its interval excludes zero.

### Step 1: overall effect (layer 1)

Removing each component changes test nDCG@10 by (variant minus full; negative = it was helping): candidate-aware attention −0.0030 [−0.0041, −0.0019], ID masking −0.0039 [−0.0054, −0.0025], T1 −0.0413 [−0.0438, −0.0389], T2 −0.0002 [−0.0014, +0.0009], T3 −0.0043 [−0.0058, −0.0028], U −0.0252 [−0.0280, −0.0222]. Every component except T2 helps overall ranking detectably; T1 and U account for most of it.

### Step 2: cold vs warm (layer 2). This is the second half of the research question

| Component | Cold Recall@10, validation | Cold Recall@10, test | Reading (test) | Warm Recall@10, test | Reading (warm) | Zero-exposure Recall@10, test |
|---|---|---|---|---|---|---|
| Candidate-aware attention | +0.0283 [+0.0145, +0.0436] | +0.0282 [+0.0137, +0.0446] | helps | +0.0047 [+0.0027, +0.0065] | helps | +0.0548 |
| ID masking | −0.0013 [−0.0194, +0.0160] | +0.0046 [−0.0139, +0.0248] | no detectable effect | +0.0069 [+0.0044, +0.0093] | helps | −0.0350 |
| T1 own behaviour | +0.0075 [−0.0166, +0.0316] | −0.0242 [−0.0509, +0.0018] | no detectable effect | +0.0392 [+0.0356, +0.0427] | helps | +0.0408 |
| T2 borrowed behaviour | +0.0122 [−0.0036, +0.0294] | +0.0073 [−0.0102, +0.0248] | no detectable effect | +0.0012 [−0.0009, +0.0033] | no detectable effect | −0.0210 |
| T3 explicit content | +0.0079 [−0.0104, +0.0254] | +0.0125 [−0.0060, +0.0313] | no detectable effect | +0.0050 [+0.0024, +0.0076] | helps | +0.0268 |
| U reader match | +0.0724 [+0.0444, +0.1004] | +0.0113 [−0.0193, +0.0473] | no detectable effect | +0.0270 [+0.0232, +0.0312] | helps | +0.0594 |

(All columns are contributions, full minus variant; positive = the component helps. Table `rq_part2_tier_contributions`. The zero-exposure column rests on 143 impressions and has no interval, so differences of a few points there, such as ID masking's −0.035, are within noise.)

### Step 3: where on the exposure axis (layer 2, finer)

Figure `fig_ablation_heatmap` breaks the change down by how often the article had been shown (0, 1–4, 5–9, 10–19, 20–49, 50+). Red cells mean removing the component lowered recall there, i.e. it helps those articles; a dot marks an interval that excludes zero. Each component helps most at: candidate-aware attention → 0 (never-shown articles); ID masking → 10–19; T1 → 20–49; T2 → 1–4; T3 → 10–19; U → 10–19. The split is clear: the candidate-aware match matters most for brand-new articles, the article's own behaviour (T1) for established ones, and the reader match (U) and explicit content (T3) for articles just leaving the cold state.

### Step 4: popularity-neutral check (layer 3)

Contribution to within-cold AUC: candidate-aware attention +0.0145 (helps), ID masking +0.0110 (no detectable effect), T1 +0.0008 (no detectable effect), T2 +0.0019 (no detectable effect), T3 +0.0145 (helps), U +0.0354 (helps). Candidate-aware attention helps cold recall *and* within-cold AUC: it finds more new articles and picks the right ones. U and T3 help within-cold AUC without a detectable recall effect: they sharpen the order among new articles rather than moving them up as a group. T1 does neither, as expected for a tier that has nothing to say about a new article.

### Step 5: hypothesis check

| Component | Expected (section 1.4) | Observed (test) | Match? |
|---|---|---|---|
| T1 | Helps warm and overall, not zero-exposure | warm: helps; cold: no detectable effect; peak at 20–49 | **Yes** (its +0.04 on never-shown articles has no interval; T1 also holds the article's age and exposure count, which are informative even at zero) |
| T2 | Helps cold (0–9 exposures) | cold: no detectable effect; warm: no detectable effect; peak at 1–4 | **No.** Borrowed behaviour is redundant once the network reads content directly |
| T3 | Helps zero-exposure and within-cold | zero: +0.027; within-cold AUC: helps | **Partly.** It helps order new articles but adds no detectable recall; the news encoder already reads the full content |
| U | Every layer, most for readers with history | overall: helps; warm: helps; cold: helps on validation, same sign on test; within-cold AUC: helps | **Mostly** |
| Candidate-aware attention | Helps cold by letting the candidate pick relevant history | cold: helps on both splits, peak at 0 exposures; attention weights uniform (section 2.6) | **Result yes, mechanism no.** It helps as a second candidate-specific match, not by selecting particular clicks |
| ID masking | Helps cold by forcing the network off ID look-ups | cold: no detectable effect; overall and warm: help | **No.** It acts as a regulariser |

### Step 6: what the ablation cannot tell us

* **Overlap.** Content enters through the news encoder in every variant, and through T3, T2 and U as engineered columns. Removing a tier removes only its engineered columns, so "no detectable effect" for T2 or T3 means "redundant given the encoders", not "content does not matter".
* **One training seed** per variant. Validation Cold Recall@10 of the full model moved by up to 0.12 between epochs, so training randomness on cold metrics may be as large as the bootstrap intervals, which cover only which readers were sampled.
* **Decision rule** (fixed before test): a component "helps cold articles" only if removing it lowers Cold Recall@10 with the validation interval below zero and the test change has the same sign (Table `ablation_decisions`): candidate-aware attention **helps cold**, U **helps cold**, ID masking, T1, T2 and T3 **no detectable effect**.

# 4. Segment diagnostics: for whom, and when?

Chapter 3 says which component matters. This chapter asks **for which articles, readers and moments** the main model gains or loses.

### Step 1: by article exposure

Figure `fig_recall_by_exposure`, Table `segment_exposure`. For never-shown articles the main model reaches Recall@10 0.7063 (best simple baseline 0.0769, NRMS 0.6352). For the most-shown (50+) articles it reaches 0.6636 (best baseline 0.5963; NRMS 0.5766). The gap over the baseline shrinks from +0.63 for never-shown articles to +0.07 for the most-shown ones: content does the work where the article has no history, and the baseline's click counts catch up once it has one. Against NRMS the gap is the other way round, small for new articles (+0.07) and larger for established ones (+0.09). Behaviour (T1) is what the content-only network lacks.

### Step 2: by reader history

Table `segment_history`. U and the candidate-aware match need a reading history.

| Reader's past clicks | Cold-click impressions | Main model | NRMS | Best baseline | Main minus baseline |
|---|---|---|---|---|---|
| 0 | 18 | 0.7778 | 0.5556 | 0.3889 | +0.39 |
| 1–4 | 60 | 0.6500 | 0.5500 | 0.2250 | +0.43 |
| 5–19 | 253 | 0.6772 | 0.5547 | 0.2266 | +0.45 |
| 20–49 | 250 | 0.5847 | 0.5467 | 0.2780 | +0.31 |
| 50+ | 256 | 0.6022 | 0.6276 | 0.2383 | +0.36 |

(Cold Recall@10, test.) The advantage over the baseline does not grow with history: the article-side evidence carries the cold gain, and personalisation adds to it rather than producing it. Main-model cold recall is highest for light readers (5–19 past clicks, 0.68) and lower for heavy ones (0.58–0.60). For the heaviest readers, NRMS is level with it (0.63 vs 0.60), a sign that the engineered features add little for readers whose long history the encoders already read well. The no-history row rests on 18 impressions and should not be read on its own.

### Step 3: by category

Tables `segment_category` and `business_category_gain` (main model minus best baseline, Cold Recall@10, test):

| Category | Gain | 95% interval | Cold-click impressions |
|---|---|---|---|
| News | +0.36 | [+0.31, +0.42] | 310 |
| Sports | +0.47 | [+0.40, +0.54] | 261 |
| Food and drink | +0.22 | [+0.04, +0.43] | 45 |
| Travel | +0.19 | [+0.03, +0.32] | 43 |
| Lifestyle | +0.29 | [+0.10, +0.49] | 41 |
| Finance | +0.45 | [+0.26, +0.63] | 40 |
| Autos | +0.33 | [+0.17, +0.50] | 29 |
| Video | +0.54 | [+0.35, +0.75] | 24 |
| Weather | +0.46 | [+0.21, +0.69] | 24 |
| Health | +0.35 | [+0.11, +0.57] | 23 |
| Music | +0.20 | [+0.05, +0.40] | 20 |
| TV | +0.10 | [+0.00, +0.33] | 10 |
| Entertainment | +0.00 | [−0.33, +0.33] | 9 |
| Movies | +0.29 | [+0.00, +0.67] | 7 |

11 of the 14 categories with clicked cold articles gain detectably, and none lose (kids has no clicked cold article on test). The two large categories, news and sports, account for two thirds of cold-click impressions and gain +36 and +47 points. The categories without a detectable gain (TV, entertainment, movies) have 10 or fewer impressions each.

### Step 4: by slate length and time of day

Tables `segment_slate` and `segment_hour`; Cold Recall@10 and its lift over random (test):

| Slate length | Cold-click impressions | Main model (lift) | NRMS (lift) | Best baseline (lift) |
|---|---|---|---|---|
| 2–9 | 22 | 1.00 (0.00) | 1.00 (0.00) | 1.00 (0.00) |
| 10–24 | 124 | 0.93 (+0.40) | 0.85 (+0.31) | 0.44 (−0.10) |
| 25–49 | 227 | 0.77 (+0.48) | 0.63 (+0.34) | 0.24 (−0.04) |
| 50–99 | 297 | 0.49 (+0.35) | 0.52 (+0.37) | 0.17 (+0.02) |
| 100+ | 167 | 0.40 (+0.32) | 0.34 (+0.26) | 0.17 (+0.09) |

In slates of 9 or fewer, every article is on the first screen, so nothing can be learned there. In every longer slate, the main model places new articles 0.32 to 0.48 above random, while the best baseline's lift is about zero: on new articles, ranking by smoothed click rate is no better than chance. The main model leads NRMS everywhere except 50–99-article slates, where the two are level (0.49 vs 0.52).

By time of day, the main model's Cold Recall@10 is steady: 0.62 (00–05h), 0.61 (06–11h), 0.66 (12–17h), 0.63 (18–23h). NRMS drops overnight (0.48 at 00–05h, when fewer readers and fresh clicks exist), and the baseline stays at 0.23–0.27, about random. There is no hour where the main model dips, so its 6-hour momentum features do not visibly lag the news cycle.

### Step 5: stress test on rebuilt slates

Every impression is rebuilt as 10 articles, 40% warm and 60% cold, keeping every click (Table `stress_slates`, three random rebuilds; 832 test impressions qualify). This is a resampled, secondary test.

| Model | nDCG@10 | Cold Recall@5 | Warm Recall@5 | Within-cold AUC | Cold-vs-all AUC |
|---|---|---|---|---|---|
| **Deep hybrid (main)** | 0.618 | **0.753** | 0.563 | **0.655** | **0.705** |
| Group 10 NRMS | 0.556 | 0.708 | 0.506 | 0.596 | 0.639 |
| Best simple baseline | 0.639 | 0.489 | 0.689 | 0.542 | 0.547 |
| Popularity | **0.715** | 0.146 | **1.000** | 0.500 | 0.342 |
| Random | 0.533 | 0.493 | 0.495 | 0.500 | 0.500 |

(Standard deviation across the three rebuilds is 0.02 or less.) When cold articles are the majority of the slate, the main model still finds the most new articles and orders them best: it leads every model on cold recall, within-cold AUC and cold-vs-all AUC. It does not lead on overall nDCG. In these rebuilt slates, the warm articles kept are the ones readers clicked, so ranking every warm article first (popularity: warm Recall@5 1.00) scores well while finding almost no new articles (0.15). On a cold-heavy screen, the main model trades some established-article clicks for new ones. This is the same trade-off layer 5 prices, but at a much higher cold share than real slates have.

# 5. Error analysis: why the model still fails on some cold articles

This chapter looks at individual cases, using validation data so the test day stays a confirmation split.

### Step 1: three kinds of case

| Case | Definition | Count |
|---|---|---|
| Success | Clicked cold article ranked in the top 3 | 356 |
| Failure | Clicked cold article ranked in the bottom quarter of its slate | 9 |
| False alarm | Cold article ranked in the top 3 but not clicked | 2,089 |

Outright failures are rare: 9 clicked cold articles in the bottom quarter against 356 in the top 3. The patterns below rest on those 9 cases, so they are indicative.

### Step 2: which evidence was missing, tier by tier

For each tier we compare its features in failures against successes (Table `error_patterns_by_tier`; a standardised difference of 0.2 or more is notable). Notable differences: T1 4, T2 4, T3 2, U 4, slate context 2. The largest:

1. failure cases have lower text similarity to the reader's history than success cases (0.108 vs 0.359) [U user matching]
2. failure cases have lower reader's affinity for the category than success cases (0.0251 vs 0.391) [U user matching]
3. failure cases have lower reader's affinity for the subcategory than success cases (0 vs 0.212) [U user matching]

All three of the largest differences are in reader match (U). The model fails on new articles that fall **outside a reader's usual interests**: the reader clicked an article in a category they had almost never read (affinity 0.03 vs 0.39), with little textual overlap with their history. This matches the ablation: U and the candidate-aware match, both of which compare the article with the reader's history, are the components that help cold articles. When the history says nothing about the article, the model has nothing to go on.

### Step 3: read the cases

`error_samples.csv` holds 30 cases of each type with titles, and all 9 failures. Three patterns hold across the failures:

* **Soft, general-interest pieces shown to readers who never read that category.** Lifestyle, food, TV and finance make up 5 of the 9 failures (they are 9% of successes). Examples: "27 bad eating habits you need to stop right now" and "50 Work-From-Home Jobs Paying as Much or a Lot More Than the Average American Salary". In 7 of 9 failures the reader's affinity for the article's category is zero or near zero.
* **Not missing content.** 8 of the 9 failures have an abstract, the same rate as successes (87%). Their exposure counts (3 to 9) and ages (mostly under 4 hours) are typical of cold articles. The model had the content; what it lacked was a reason to show it to *this* reader.
* **Long slates.** Two failures are impeachment stories in 165-article slates, shown to readers with little news history. With that many candidates, any article that does not match the reader's profile lands far down.

Successes and false alarms look alike (mostly news and sports, category affinity 0.31 and 0.37, text similarity 0.35 and 0.33). The model's top cold picks are the articles that fit the reader's habits, whether or not they get clicked.

### Step 4: what it means for the model

The model lacks a signal for *how exploratory a reader is*. It ranks new articles by how well they fit the reader's past, so readers who sometimes click outside their habits are where it fails. A reader novelty-propensity feature (how often the reader has clicked outside their top categories before) would supply it. So would a small exploration quota, which layer 5 shows is cheap.

# 6. Answer to the research question

> *Does combining T1, T2 and T3 improve personalised ranking for under-exposed articles, and how does each tier's contribution differ between cold and warm articles?*

**Part 1: does combining help?** Against the best simple baseline (behaviour only) the main model is **better for cold articles and better for warm articles**: of every 100 clicked cold articles, 63 were on the first screen, against 25; for warm articles 67 against 60. Against Group 10's NRMS (content only) it is also **better for cold articles and better for warm articles** (Table `rq_part1_combination`).

**Part 2: how do the tiers differ?**

* **T1 (own behaviour):** on cold articles **no detectable effect**, on warm articles **helps**; matters most at 20–49 prior exposures.
* **T2 (borrowed behaviour):** on cold articles **no detectable effect**, on warm articles **no detectable effect**.
* **T3 (explicit content):** on cold articles **no detectable effect** on recall but **helps** within-cold ordering; on warm articles **helps**; matters most at 10–19. Content also reaches the model through the news encoder, which every variant keeps.
* **U (reader match):** on cold articles **helps under the decision rule** (validation +0.072, test same sign but not detectable) and **helps** within-cold ordering; on warm articles **helps**.
* **Candidate-aware match:** on cold articles **helps** (both splits; most for never-shown articles), on warm articles **helps** slightly.

**One-paragraph answer.** Yes. Combining a content-and-history network with behavioural and matching features raises Cold Recall@10 by 0.38 over the behaviour-only baseline and by 0.05 over the content-only network. Warm recall rises too (+0.07 and +0.09), so new articles gain at no cost to established ones. The tiers do different jobs. Own behaviour (T1) carries the warm ranking and does nothing detectable for cold articles. New articles are placed by reading their content and matching it to the reader: the news encoder, the candidate-aware match and the reader–article features (U), with U and T3 sharpening which new articles go first. Borrowed behaviour (T2) adds nothing once content is read directly. The result holds under popularity-neutral metrics (within-cold AUC 0.65 vs 0.55; popularity-debiased nDCG 0.42 vs 0.34), against the baseline at thresholds 5 and 20, and when cold articles are the majority of the slate. Against the content-only network, the cold gain holds at thresholds 10 and 20 but is not detectable at 5 on test, where articles have almost no behaviour for the engineered features to use.

# 7. Business insights

## 7.1 What the numbers mean in plain language

| Technical result (test) | Plain-language meaning |
|---|---|
| Cold Recall@10 = 0.6268 | Of every 100 times a reader clicked a **new** article, the ranker had put it on the first screen 63 times (simple ranker 25, popularity 3) |
| Warm Recall@10 = 0.6660 | Same for **established** articles: 67 in 100 (simple ranker 60) |
| Within-cold AUC = 0.6475 | Given two new articles, one clicked and one not, the ranker puts the clicked one higher 65% of the time (a coin flip is 50%) |
| Least-shown 80% get 3.9% of first-screen slots | More than under any other ranker (random top 10: 2.3%) and more than their 3.3% share of MSN's logged impressions |
| +11.4 clicked cold articles in the top 10 per 1,000 visits [+10.1, +12.7] | Per 1,000 page views, 11 more wanted new articles are on the first screen than with the best simple ranker, on top of 104 more wanted established ones |

## 7.2 How a business could use this

| If the goal is … | Then … | Evidence |
|---|---|---|
| More engagement with fresh content | Use the main model; add at most a 1–2 slot quota for new articles | Layer 2 and the quota table |
| Fair exposure for new publishers or creators | Track "share of first-screen slots for new articles" and "for the least-shown 80%" as separate KPIs | Layer 4 |
| Protect core engagement | Cap the quota where warm loss passes the agreed tolerance | Layer 5, balance rule |
| Decide where to spend data-engineering effort | Real-time article counters, entity tagging and clean reading histories; category-level CTR feeds can be dropped | Ablation, chapter 3 |
| Readers who explore outside their habits | Expect misses on new articles outside their usual categories; a small exploration quota helps | Error analysis, chapter 5 |

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
* Never-shown articles (143 test impressions), readers with no history (18), small categories (under 30 impressions each) and the 9 error-analysis failures rest on little data and are indicative only.
* Against NRMS, the cold gain is not detectable on test at the strictest cold threshold (fewer than 5 exposures).
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
