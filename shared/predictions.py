"""Prediction files handed from the model notebooks to the evaluator.

Format (handover B4): one row per candidate with columns
row_id, model_id, config_id, predicted_score. Labels, impression ids and cold
tags are joined later from meta_{split}.parquet on row_id, never stored here.

Each file records its data kind three ways: the predictions/real/ folder, a
REAL__ filename prefix, and a data_kind entry in the parquet metadata.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PRED_COLUMNS = ["row_id", "model_id", "config_id", "predicted_score"]
DATA_KINDS = ("REAL",)


def prediction_path(root: str, data_kind: str, model_id: str, config_id: str, split: str) -> str:
    assert data_kind in DATA_KINDS, data_kind
    safe_cfg = config_id.replace("=", "-").replace(",", "_").replace(" ", "")
    return os.path.join(root, data_kind.lower(), f"{data_kind}__{model_id}__{safe_cfg}__{split}.parquet")


def write_predictions(row_id, score, *, root: str, data_kind: str, model_id: str, config_id: str,
                      split: str, schema_version: str, extra: dict | None = None) -> str:
    """Validate and write one prediction file. Returns its path."""
    row_id = np.asarray(row_id, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    assert len(row_id) == len(score), "row_id and score lengths differ"
    assert len(np.unique(row_id)) == len(row_id), "duplicate row_id in predictions"
    assert np.isfinite(score).all(), "non-finite predicted_score"
    df = pd.DataFrame({"row_id": row_id, "model_id": model_id, "config_id": config_id, "predicted_score": score})
    table = pa.Table.from_pandas(df[PRED_COLUMNS], preserve_index=False)
    info = {"data_kind": data_kind, "model_id": model_id, "config_id": config_id, "split": split,
            "schema_version": schema_version, "n_rows": int(len(df)),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"g11": json.dumps(info).encode()})
    path = prediction_path(root, data_kind, model_id, config_id, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)
    return path


def read_predictions(path: str) -> tuple[pd.DataFrame, dict]:
    table = pq.read_table(path)
    info = json.loads((table.schema.metadata or {}).get(b"g11", b"{}"))
    return table.to_pandas(), info
