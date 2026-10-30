"""Validate two clustering features and non-identifying audit IDs."""

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["annual_income_k", "spending_score"]


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Features must be annual_income_k and spending_score only")
    result = frame[FEATURES].copy()
    for name in FEATURES:
        result[name] = pd.to_numeric(result[name], errors="raise")
        if not np.isfinite(result[name].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must contain finite numbers without missing values")
    if not result.annual_income_k.between(0, 1000, inclusive="right").all():
        raise ValueError("annual_income_k must be greater than 0 and at most 1000")
    if not result.spending_score.between(1, 100).all() or (result.spending_score % 1 != 0).any():
        raise ValueError("spending_score must be a whole number from 1 to 100")
    return result


def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if set(raw.columns) != {"customer_id", *FEATURES}:
        raise ValueError("CSV must contain exactly customer_id, annual_income_k, spending_score")
    if len(raw) < 20:
        raise ValueError("Use at least 20 customer observations")
    identifiers = raw.customer_id.astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("customer_id must be unique and nonblank")
    result = validate_features(raw[FEATURES])
    if len(result.drop_duplicates()) < 10 or (result.nunique() < 2).any():
        raise ValueError("Use at least 10 distinct feature pairs with variation in both features")
    result.insert(0, "customer_id", identifiers)
    return result
