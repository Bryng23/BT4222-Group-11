# BT4222 Group 11: Cold-start news recommendation on MIND

Does combining candidate behaviour, content-transferred behaviour and pure content improve personalised ranking for under-exposed articles? Dataset: MIND-small **train** split (9–14 Nov 2019), split by time into train (Nov 9–12), validation (Nov 13) and test (Nov 14). An article is *cold* when it has fewer than 10 prior exposures at impression time.

**Main model:** a deep ranker built on Group 10's NRMS-inspired attention network, extended with an engineered-feature tower, candidate-aware attention over the reader's history, a GMF term and an article ID embedding with ID masking (step 4). **Final numbers:** step 6 only; steps 3–5 report interim metrics used for tuning.

## Layout

| Path | Purpose |
|---|---|
| `step1_data_exploration.ipynb` | Exploratory analysis behind the interim report: slates, exposure concentration, click and feedback patterns, categories, the CTR-noise curve behind the cold threshold; revised findings are marked |
| `step2_feature_engineering.ipynb` | Preprocessing, temporal split, cold labels and every engineered feature; exports the tables in schema contract v0.1 |
| `step3a_simple_baselines.ipynb` | Random, popularity, category CTR, candidate and content smoothed CTR; controlled smoothing comparison |
| `step3b_nrms_g10.ipynb` | Group 10's NRMS-inspired model (the past-project baseline), run on our split (GPU runtime) |
| `step4_deep_hybrid.ipynb` | **Main model**: training, scoring, comparison with every baseline, attention read-out (GPU runtime) |
| `step5_deep_ablation.ipynb` | Ablations of the main model, resumable across Colab sessions (GPU runtime) |
| `step6_evaluation_diagnostics.ipynb` | Final evaluation: headline table, cold/warm/never-shown recall, robustness, exposure, balance, ablations, segments, error analysis, business numbers; fills the report sections (no training) |
| `shared/schema.py` | Contract constants, column dtypes, file layout |
| `shared/feature_groups.py` | Feature-group (tier) membership |
| `shared/contract_checks.py` | Checks the exported feature tables against the contract |
| `shared/derived.py` | Smoothed-CTR scores computed from stored columns |
| `shared/predictions.py` | Writes and reads prediction files (`row_id, model_id, config_id, predicted_score`) |
| `shared/selection.py` | Validation selection rule shared by the tuned baselines |
| `shared/eval_interim.py` | Interim evaluator used for tuning in steps 3–5 |
| `shared/nrms_inputs.py`, `shared/nrms_g10.py` | Inputs and model for Group 10's network |
| `shared/hybrid_model.py` | Main model: feature tower, candidate-aware attention, GMF term, ID masking, training loop, attention read-out |
| `shared/evaluator.py` | Team evaluator: MIND metrics, cold/warm/never-shown recall, cold AUCs, popularity-debiased nDCG, slot share, coverage, user-clustered bootstrap |
| `shared/segments.py`, `shared/audit.py`, `shared/diagnostics.py` | Segments; independent data and file checks; quota re-ranking, balance rule, stress slates, error analysis, business numbers, ablation helpers |
| `shared/report_plots.py`, `shared/report_io.py` | Figure templates; table export and report-template filling |
| `tests/test_evaluator.py` | Every metric checked against a brute-force implementation |
| `docs/evaluation_spec.md` | Evaluation rules step 6 implements |
| `docs/report_sections_template.md`, `docs/report_sections_filled.md` | Report sections; step 6 fills every number |

**Run order:** 1 (exploration only; nothing depends on it) → 2 → 3a, 3b → 4 → 5 → 6. Steps 4 and 5 reuse the text-vector cache step 3b writes; step 5 compares against step 4's predictions; step 6 reads every prediction file. Each notebook opens with an *At a glance* table: what it reads and writes, how long it takes, and what must run first.

### Supplementary (not part of the report)

A LightGBM ranker was explored before the main model became a deep network. It is kept for reference and is not used in the report's comparisons or in step 6.

| Path | Purpose |
|---|---|
| `step3c_lightgbm.ipynb` | LightGBM (LambdaRank) ranker on the same features, with a binary-objective comparison |
| `step3d_lightgbm_ablation.ipynb` | Its feature-group ablations; reads the tree settings step 3c saves |
| `shared/lgbm_pipeline.py` | Train-only PCA, design matrices, LightGBM training |

## Data

The raw MIND-small files are downloaded automatically by `kagglehub` (`arashnic/mind-news-dataset`). Everything the notebooks produce is too large for GitHub (over 100 MB), so the intermediate data is shared on Google Drive:

**Intermediate data:** [BT4222_G11 folder on Google Drive](https://drive.google.com/drive/folders/1OHrQA62pxxHQHGb0GYHpI89yGB-FLih0)

The notebooks read and write this folder at `MyDrive/BT4222_G11/`:

| Path | Contents | Written by | Read by |
|---|---|---|---|
| `features/v0.1/` | Per-split meta and feature tables (`meta_*`, `features_*`), article and user vectors, category and subcategory maps, `schema_version.json` | step 2 | steps 3a–6 |
| `nrms_cache/real/` | MiniLM title-and-abstract vectors and entity vectors (`.npz`) | step 3b | steps 3b, 4, 5 |
| `predictions/real/` | One file per model, configuration and split, named `REAL__<model>__<config>__<split>.parquet`, with columns `row_id, model_id, config_id, predicted_score` | steps 3a–5 | steps 4–6 |
| `models/real/` | Training histories (validation metrics per epoch) of the step 5 ablation variants | step 5 | reference only |
| `step6_outputs/real/` | Final evaluation: `tables/`, `figures/`, `report_numbers.json`, `report_sections_filled.md` | step 6 | the report |
| `Data.zip` | Snapshot of `features/`, `nrms_cache/` and `predictions/` taken before the step 5 ablations; the data-setup cell in step 5 unpacks it | — | step 5 |
| `BT4222_Group11_Github_repo_notebooks.zip` | Copy of this repository's notebooks | — | — |

To run the notebooks in Colab, the folder must sit at `MyDrive/BT4222_G11/` in your own Drive with edit access, because every step from 2 to 6 writes into it. Download the shared folder and upload it to your My Drive as `BT4222_G11`. To re-run from step 5 only, uploading `Data.zip` and running step 5's data-setup cell is enough. Open a notebook from GitHub in Colab (File → Open notebook → GitHub), choose a T4 GPU runtime for steps 3b, 4 and 5, and run all.
