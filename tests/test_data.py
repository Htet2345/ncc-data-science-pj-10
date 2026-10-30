import numpy as np
import pandas as pd
import pytest

from mall_clustering.data import FEATURES, load_data, validate_features
from mall_clustering.model import DEFAULT_DATA
from scripts.generate_data import generate


def test_synthetic_data_is_reproducible(tmp_path):
    frame = load_data(DEFAULT_DATA)
    assert frame.shape == (200, 3)
    assert frame.customer_id.str.startswith("DEMO").all()
    path = tmp_path / "customers.csv"
    generate(path)
    pd.testing.assert_frame_equal(load_data(path), frame)


@pytest.mark.parametrize("kind", ["missing_column", "extra_column", "few_rows", "nan", "inf", "zero_income", "large_income", "low_score", "high_score", "fractional_score", "duplicate_id", "blank_id", "no_variation", "few_unique"])
def test_invalid_data(tmp_path, kind):
    frame = pd.read_csv(DEFAULT_DATA)
    if kind == "missing_column":
        frame = frame.drop(columns="spending_score")
    elif kind == "extra_column":
        frame["unused"] = 0
    elif kind == "few_rows":
        frame = frame.head(10)
    elif kind == "nan":
        frame.loc[0, "annual_income_k"] = np.nan
    elif kind == "inf":
        frame.loc[0, "annual_income_k"] = np.inf
    elif kind == "zero_income":
        frame.loc[0, "annual_income_k"] = 0
    elif kind == "large_income":
        frame.loc[0, "annual_income_k"] = 1001
    elif kind == "low_score":
        frame.loc[0, "spending_score"] = 0
    elif kind == "high_score":
        frame.loc[0, "spending_score"] = 101
    elif kind == "fractional_score":
        frame["spending_score"] = frame.spending_score.astype(float)
        frame.loc[0, "spending_score"] = 20.5
    elif kind == "duplicate_id":
        frame.loc[1, "customer_id"] = frame.loc[0, "customer_id"]
    elif kind == "blank_id":
        frame.loc[0, "customer_id"] = " "
    elif kind == "no_variation":
        frame["annual_income_k"] = 50
    elif kind == "few_unique":
        frame["annual_income_k"] = [20 + index % 2 for index in range(len(frame))]
        frame["spending_score"] = [40 + index % 2 for index in range(len(frame))]
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_feature_order_is_restored():
    frame = load_data(DEFAULT_DATA)[FEATURES].iloc[[0]][list(reversed(FEATURES))]
    assert list(validate_features(frame).columns) == FEATURES
