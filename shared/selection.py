"""Model-selection rule shared by every notebook that tunes on validation.

Among candidate settings whose validation nDCG@10 is within NDCG_TOLERANCE of the
best one, pick the highest validation Cold Recall@10; break ties on nDCG@10.
The tolerance is provisional until the team agrees a value.
"""
from __future__ import annotations

import pandas as pd

PRIMARY = "cold_recall@10"
GUARD = "ndcg@10"
NDCG_TOLERANCE = 0.005


def select(results: pd.DataFrame, primary: str = PRIMARY, guard: str = GUARD, tolerance: float = NDCG_TOLERANCE):
    """Return the index label of the selected row of a validation results table."""
    eligible = results[results[guard] >= results[guard].max() - tolerance]
    return eligible.sort_values([primary, guard], ascending=False).index[0]


def describe() -> str:
    return (f"highest validation {PRIMARY} among settings whose validation {GUARD} is within "
            f"{NDCG_TOLERANCE} of the best; ties broken on {GUARD}")
