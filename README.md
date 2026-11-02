# BT4222 Group 11: Cold-start news recommendation on MIND

Does combining candidate behaviour, content-transferred behaviour and pure content improve personalised ranking for under-exposed articles? Dataset: MIND-small **train** split (9–14 Nov 2019), split by time into train (Nov 9–12), validation (Nov 13) and test (Nov 14).

## Layout

| Path | Owner | Purpose |
|---|---|---|
| `step1_data_preprocess.ipynb` | Andrew | Interaction table, temporal split, cold/warm labels |
| `step2_feature_engineering.ipynb` | Andrew | Feature tables per schema contract v0.1 |
| `step3a_simple_baselines.ipynb` | Bryan | Random, popularity, category CTR, candidate and content smoothed CTR; controlled smoothing comparison |
| `step3b_nrms_g10.ipynb` | Bryan | Group 10's NRMS-inspired model, corrected reimplementation (GPU runtime) |
| `step4_advanced_model.ipynb` | Bryan | LightGBM hybrid model: tuning, scoring, baseline comparison, LambdaRank comparison |
| `step5_ablation_study.ipynb` | Bryan / Shi Ting | Feature-group ablations, each retrained from scratch |
| `step6_evaluation_diagnostics.ipynb` | Shi Ting | Metrics, cold/warm segments, diagnostics |
| `shared/schema.py` | all | Contract constants, column dtypes, file layout |
| `shared/feature_groups.py` | all | Ablation group membership |
| `shared/derived.py` | Bryan | Smoothed-CTR scores computed from stored columns (β and α tuned without regenerating features) |
| `shared/predictions.py` | Bryan | Writes and reads prediction files (`row_id, model_id, config_id, predicted_score`) |
| `shared/selection.py` | all | Validation selection rule shared by every tuned model |
| `shared/lgbm_pipeline.py` | Bryan | Train-only PCA, design matrices, LightGBM training |
| `shared/nrms_inputs.py`, `shared/nrms_g10.py` | Bryan | Inputs and model for the Group 10 reimplementation |

Run order: steps 1–2 (features) → 3a, 3b → 4 → 5 → 6. Step 5 reads the tree settings that step 4 saves on Drive.

## Running in Colab

1. Open a notebook from GitHub in Colab (File → Open notebook → GitHub).
2. Runtime → Run all. The first cell mounts Google Drive and clones this repo.
3. Data and feature tables live on Google Drive under `MyDrive/BT4222_G11/` (GitHub rejects files over 100 MB). Share links: _to be added_.

Free-tier Colab is enough. Keep feature matrices in float32.
