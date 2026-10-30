"""Compare K-means fits using internal clustering diagnostics."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mall_clustering.data import FEATURES, load_data, validate_features

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "sample_data" / "customers.csv"
DEFAULT_OUTPUT = ROOT / "artifacts"


def make_pipeline(k: int) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=42, max_iter=300))])


def train(data_path: Path = DEFAULT_DATA, output_dir: Path = DEFAULT_OUTPUT, k: str | int = "auto") -> dict:
    if k != "auto" and (isinstance(k, bool) or str(k) not in [str(value) for value in range(2, 9)]):
        raise ValueError("K must be auto or an integer from 2 to 8")
    frame = load_data(data_path)
    features = frame[FEATURES]
    candidates, evaluations = {}, []
    with threadpool_limits(limits=1):
        for value in range(1, 9):
            pipeline = make_pipeline(value).fit(features)
            model = pipeline.named_steps["kmeans"]
            transformed = pipeline.named_steps["scale"].transform(features)
            score = float(silhouette_score(transformed, model.labels_)) if value > 1 else None
            evaluations.append({"k": value, "inertia": float(model.inertia_), "silhouette": score})
            candidates[value] = pipeline
    selected = max(evaluations[1:], key=lambda result: (result["silhouette"], -result["k"]))["k"] if k == "auto" else int(k)
    pipeline = candidates[selected]
    model = pipeline.named_steps["kmeans"]
    labels = model.labels_
    centers = pipeline.named_steps["scale"].inverse_transform(model.cluster_centers_)
    centroids = pd.DataFrame(centers, columns=FEATURES)
    centroids.insert(0, "cluster_id", range(selected))
    centroids["customer_count"] = [int(np.sum(labels == index)) for index in range(selected)]
    chosen = evaluations[selected - 1]
    metrics = {
        "dataset": str(data_path.resolve()), "data_note": "Bundled customers are synthetic; clustering diagnostics are descriptive, not classification accuracy.",
        "rows": len(frame), "features": FEATURES, "selected_k": selected,
        "selection": "Highest silhouette among K=2..8 on the fitted data" if k == "auto" else "User-specified K; candidate diagnostics are still exported",
        "silhouette": chosen["silhouette"], "inertia": chosen["inertia"],
        "random_state": 42, "n_init": 10,
        "cluster_counts": {str(index): int(np.sum(labels == index)) for index in range(selected)},
        "training_ranges": {name: [float(features[name].min()), float(features[name].max())] for name in FEATURES},
        "evaluations": evaluations, "scikit_learn_version": sklearn.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "schema_version": 1}, output_dir / "model.joblib")
    assigned = frame.copy()
    assigned["cluster_id"] = labels
    assigned.to_csv(output_dir / "clustered_customers.csv", index=False)
    centroids.to_csv(output_dir / "centroids.csv", index=False)
    pd.DataFrame(evaluations).to_csv(output_dir / "k_evaluation.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    from mall_clustering.plots import save_plots
    save_plots(assigned, centroids, evaluations, selected, output_dir)
    return metrics


def assign(income: float, spending_score: int, model_path: Path = DEFAULT_OUTPUT / "model.joblib") -> dict:
    row = validate_features(pd.DataFrame([[income, spending_score]], columns=FEATURES))
    # Load only trusted artifacts; joblib uses Python pickle.
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported model artifact; retrain this project")
    pipeline = bundle["pipeline"]
    with threadpool_limits(limits=1):
        cluster = int(pipeline.predict(row)[0])
        distances = pipeline.transform(row)[0]
    center = pipeline.named_steps["scale"].inverse_transform(pipeline.named_steps["kmeans"].cluster_centers_[[cluster]])[0]
    outside = [name for name, (low, high) in bundle["metrics"]["training_ranges"].items() if not low <= float(row.iloc[0][name]) <= high]
    return {
        "cluster_id": cluster,
        "centroid": dict(zip(FEATURES, map(float, center))),
        "distance_to_centroid_scaled": float(distances[cluster]),
        "outside_training_range": outside,
        "note": "Cluster IDs are arbitrary group labels, not rankings, customer value or probabilities.",
    }
