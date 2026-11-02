"""Scores derived in the pipeline from stored contract columns (contract v0.1,
"related_ctr, hybrid and baseline support").

Computing these here, rather than storing them, means alpha and beta can be
tuned on validation without regenerating any feature table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smoothed_ctr(clicks, exposures, prior, strength: float) -> np.ndarray:
    """(clicks + strength * prior) / (exposures + strength). Defined for zero exposures when strength > 0."""
    assert strength > 0, "smoothing strength must be positive"
    return ((np.asarray(clicks, np.float64) + strength * np.asarray(prior, np.float64))
            / (np.asarray(exposures, np.float64) + strength)).astype(np.float32)


def candidate_smoothed_ctr(feat: pd.DataFrame, beta: float) -> np.ndarray:
    """Baseline: candidate CTR shrunk toward the global prior CTR g_t."""
    return smoothed_ctr(feat["prior_clicks"], feat["prior_exposures"], feat["global_prior_ctr"], beta)


def content_smoothed_ctr(feat: pd.DataFrame, alpha: float) -> np.ndarray:
    """Hybrid feature H: candidate CTR shrunk toward related_ctr (subcategory -> category -> global)."""
    return smoothed_ctr(feat["prior_clicks"], feat["prior_exposures"], feat["related_ctr"], alpha)
