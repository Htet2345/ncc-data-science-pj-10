import json

import joblib
import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageChops
from sklearn.metrics import silhouette_score

from mall_clustering.data import FEATURES, load_data
from mall_clustering.model import DEFAULT_DATA, assign, train


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    output = tmp_path_factory.mktemp("mall-model")
    return output, train(output_dir=output)


def test_candidate_selection_and_metrics(fitted):
    output, metrics = fitted
    assert metrics["rows"] == 200
    assert metrics["selected_k"] == 5  # Five separated groups in the synthetic fixture.
    evaluation = pd.read_csv(output / "k_evaluation.csv")
    assert list(evaluation.k) == list(range(1, 9))
    assert pd.isna(evaluation.iloc[0].silhouette)
    winner = evaluation.iloc[1:].sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
    assert metrics["selected_k"] == int(winner.k)
    assert metrics["silhouette"] == pytest.approx(winner.silhouette)
    assert sum(metrics["cluster_counts"].values()) == 200
    assert json.loads((output / "metrics.json").read_text()) == metrics


def test_scaling_labels_inertia_and_centroids(fitted):
    output, metrics = fitted
    frame = load_data(DEFAULT_DATA)
    pipeline = joblib.load(output / "model.joblib")["pipeline"]
    assert np.allclose(pipeline.named_steps["scale"].mean_, frame[FEATURES].mean())
    assert pipeline.named_steps["scale"].n_samples_seen_ == 200
    assigned = pd.read_csv(output / "clustered_customers.csv")
    assert list(assigned.customer_id) == list(frame.customer_id)
    labels = assigned.cluster_id.to_numpy()
    assert np.array_equal(pipeline.predict(frame[FEATURES]), labels)
    scaled = pipeline.named_steps["scale"].transform(frame[FEATURES])
    centers = pipeline.named_steps["kmeans"].cluster_centers_
    inertia = ((scaled - centers[labels]) ** 2).sum()
    assert metrics["inertia"] == pytest.approx(inertia)
    assert metrics["silhouette"] == pytest.approx(silhouette_score(scaled, labels))
    exported = pd.read_csv(output / "centroids.csv").set_index("cluster_id")
    for cluster in range(metrics["selected_k"]):
        members = assigned[assigned.cluster_id == cluster]
        assert exported.loc[cluster, "customer_count"] == len(members)
        assert np.allclose(exported.loc[cluster, FEATURES].to_numpy(dtype=float), members[FEATURES].mean())


def test_saved_assignment_matches_nearest_center(fitted):
    output, _ = fitted
    result = assign(60, 50, output / "model.joblib")
    pipeline = joblib.load(output / "model.joblib")["pipeline"]
    row = pd.DataFrame([[60, 50]], columns=FEATURES)
    distances = pipeline.transform(row)[0]
    assert result["cluster_id"] == int(np.argmin(distances))
    assert result["distance_to_centroid_scaled"] == pytest.approx(distances.min())
    assert abs(result["centroid"]["annual_income_k"] - 60) < 5
    assert abs(result["centroid"]["spending_score"] - 50) < 5
    assert result["outside_training_range"] == []


def test_user_specified_k(tmp_path):
    metrics = train(output_dir=tmp_path, k=3)
    assert metrics["selected_k"] == 3
    assert len(metrics["cluster_counts"]) == 3


@pytest.mark.parametrize("k", [0, 1, 9, "invalid", True, 3.5])
def test_invalid_k(tmp_path, k):
    with pytest.raises(ValueError):
        train(output_dir=tmp_path / "unused", k=k)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("income,score", [(0, 50), (float("nan"), 50), (float("inf"), 50), (60, 0), (60, 101), (60, 50.5)])
def test_invalid_customer_input(fitted, income, score):
    with pytest.raises(ValueError):
        assign(income, score, fitted[0] / "model.joblib")


def test_out_of_training_range(fitted):
    result = assign(900, 100, fitted[0] / "model.joblib")
    assert "annual_income_k" in result["outside_training_range"]


def test_plot_images(fitted):
    for name in ("clusters.png", "elbow.png", "silhouette.png"):
        with Image.open(fitted[0] / name) as image:
            assert image.format == "PNG" and image.size == (1600, 960)
            assert ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()
