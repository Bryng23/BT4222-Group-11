# BT4222 Group 11: Cold-start news recommendation on MIND

Does combining candidate behaviour, content-transferred behaviour and pure content improve personalised ranking for under-exposed articles? Dataset: MIND-small **train** split (9–14 Nov 2019), split by time into train (Nov 9–12), validation (Nov 13) and test (Nov 14).

## Layout

| Path | Owner | Purpose |
|---|---|---|
| `step1_data_preprocess.ipynb` | Andrew | Interaction table, temporal split, cold/warm labels |
| `step2_feature_engineering.ipynb` | Andrew | Feature tables per schema contract v0.1 |
| `step3_baselines.ipynb` | Bryan | Popularity, smoothed CTR, category CTR, Group 10's NRMS-inspired model (corrected reimplementation) |
| `step4_advanced_model.ipynb` | Bryan | LightGBM hybrid model |
| `step5_ablation_study.ipynb` | Bryan / Shi Ting | Tier ablations |
| `step6_evaluation_diagnostics.ipynb` | Shi Ting | Metrics, cold/warm segments, diagnostics |
| `shared/schema.py` | all | Contract constants, column dtypes, file layout |
| `shared/feature_groups.py` | all | Ablation group membership |

## Running in Colab

1. Open a notebook from GitHub in Colab (File → Open notebook → GitHub).
2. Runtime → Run all. The first cell mounts Google Drive and clones this repo.
3. Data and feature tables live on Google Drive under `MyDrive/BT4222_G11/` (GitHub rejects files over 100 MB). Share links: _to be added_.

Free-tier Colab is enough. Keep feature matrices in float32.
