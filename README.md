# BT4222 Group 11: Cold-start news recommendation on MIND

Does combining candidate behaviour, content-transferred behaviour and pure content improve personalised ranking for under-exposed articles? Dataset: MIND-small **train** split (9–14 Nov 2019), split by time into train (Nov 9–12), validation (Nov 13) and test (Nov 14).

## Layout

| Path | Owner | Purpose |
|---|---|---|
| `step1_data_preprocess.ipynb` | Andrew | Interaction table, temporal split, cold/warm labels |
| `step2_feature_engineering.ipynb` | Andrew | Feature tables per schema contract v0.1 |
| `step3a_simple_baselines.ipynb` | Bryan | Random, popularity, category CTR, candidate and content smoothed CTR; controlled smoothing comparison |
| `step3b_nrms_g10.ipynb` | Bryan | Group 10's NRMS-inspired model, corrected reimplementation (GPU runtime) |
| `step3c_lightgbm.ipynb` | Bryan | LightGBM (LambdaRank), the strongest non-deep comparison: tuning, scoring, binary-objective comparison |
| `step3d_lightgbm_ablation.ipynb` | Bryan | LightGBM feature-group ablations, each retrained from scratch |
| `step4_deep_hybrid.ipynb` | Bryan | **Main model**: Group 10's attention network + engineered-feature tower + ID embedding with ID masking (GPU runtime) |
| `step5_deep_ablation.ipynb` | Bryan / Shi Ting | Ablations of the main model, resumable across Colab sessions (GPU runtime) |
| `step6_evaluation_diagnostics.ipynb` | Shi Ting | Final evaluation of the main model against the baselines and NRMS, its ablations, segments, error analysis and business numbers; fills the report sections (no training) |
| `shared/schema.py` | all | Contract constants, column dtypes, file layout |
| `shared/feature_groups.py` | all | Ablation group membership |
| `shared/derived.py` | Bryan | Smoothed-CTR scores computed from stored columns (β and α tuned without regenerating features) |
| `shared/predictions.py` | Bryan | Writes and reads prediction files (`row_id, model_id, config_id, predicted_score`) |
| `shared/selection.py` | all | Validation selection rule shared by every tuned model |
| `shared/lgbm_pipeline.py` | Bryan | Train-only PCA, design matrices, LightGBM training |
| `shared/nrms_inputs.py`, `shared/nrms_g10.py` | Bryan | Inputs and model for the Group 10 reimplementation |
| `shared/hybrid_model.py` | Bryan | Main deep model: feature tower, GMF term, ID masking, training loop, attention read-out |
| `shared/evaluator.py` | Shi Ting | Team evaluator: MIND metrics, cold/warm/zero-exposure recall, cold AUCs, cold-only and popularity-debiased nDCG, slot share, novelty, coverage, lift over random, user-clustered bootstrap |
| `shared/segments.py` | Shi Ting | Exposure / category / history / slate / hour segments; `evaluate_by(meta, score, by=...)` |
| `shared/audit.py` | Shi Ting | Independent checks: recomputed cold flags, prediction-file alignment, cold-click support, threshold justification |
| `shared/diagnostics.py` | Shi Ting | Slot-quota re-ranking, balance rule, stress slates, error analysis, business numbers, ablation helpers |
| `shared/report_plots.py`, `shared/report_io.py` | Shi Ting | Figure templates; table export and report-template filling |
| `tests/test_evaluator.py` | Shi Ting | Every metric checked against a brute-force implementation (`python -m pytest tests -q`) |
| `docs/evaluation_spec.md` | Shi Ting | Evaluation rules step 6 implements |
| `docs/report_sections_template.md` | Shi Ting | Report sections; step 6 fills every `{{number}}` and writes `step6_outputs/real/report_sections_filled.md` on Drive |

Run order: steps 1–2 (features) → 3a, 3b, 3c → 3d → 4 → 5 → 6. Step 3d reads the tree settings step 3c saves; step 4 reuses the MiniLM/spaCy cache step 3b writes; step 5 compares against step 4's predictions; step 6 reads every prediction file and trains nothing.

## Running in Colab

1. Open a notebook from GitHub in Colab (File → Open notebook → GitHub).
2. Runtime → Run all. The first cell mounts Google Drive and clones this repo.
3. Data and feature tables live on Google Drive under `MyDrive/BT4222_G11/` (GitHub rejects files over 100 MB). Share links: _to be added_.

Free-tier Colab is enough. Keep feature matrices in float32.
