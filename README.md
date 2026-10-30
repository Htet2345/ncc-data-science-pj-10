# Mall Customer Clustering - Data Science Project 10

Exploratory customer grouping with K-means, pandas and scikit-learn. The topic
matches Module 5.4, **video 33: Mall Customer Clustering**; videos 30-32 introduce
unsupervised learning, K-means and plotting.

This is an original implementation based on those lesson titles, not a verified
copy of the teacher's exact code. The bundled CSV has 200 reproducible synthetic
customers; no real customer data is included. Its generator and provenance are
documented under `sample_data/`.

## How this differs from classification

There are no known outcome labels to predict. K-means assigns observations to
nearby centers in standardized income/spending space. This project reports
**inertia and silhouette score**, not classification accuracy. Cluster IDs are
arbitrary labels, not rankings or judgements about customers.

## Workflow

1. Load and validate income, spending score and unique audit IDs.
2. Standardize the two features so their numeric scales do not dominate distance.
3. Fit candidate K-means models for K=1..8, with seed 42 and 10 initializations.
4. Record inertia for every K and silhouette for K=2..8.
5. Select the highest-silhouette candidate by default, or use a supplied K.
6. Export row assignments, original-unit centroids, summaries and diagnostic plots.
7. Save the fitted pipeline and assign a fictional new point to its nearest center.

This is a descriptive fit to all input observations. No supervised train/test
accuracy is claimed. Both scaling and K selection use the fitted dataset;
the same-data silhouette is an internal diagnostic, not an independent validation
score. The synthetic generator's group labels are not stored or supplied to K-means.

## Inputs

| Column | Meaning | Validation |
| --- | --- | --- |
| customer_id | Arbitrary audit ID; excluded from clustering | Unique and nonblank |
| annual_income_k | Annual income in thousands of synthetic USD | Greater than 0, at most 1000 |
| spending_score | Invented spending score | Whole number 1-100 |

For example, income `60` means 60 thousand synthetic USD, not 60 dollars.
Values must be finite and present. A dataset needs at least 20 rows, 10 distinct
feature pairs and variation in both features. The broad bounds are input guards,
not evidence of training coverage. Prediction output flags numeric values outside
the observed ranges; a point far from every center still receives a nearest-center
assignment. The program is not an outlier detector.

## Setup on Windows

Requires Python 3.10+; developed with Python 3.13.

Open the folder in VS Code and choose **Terminal > New Terminal**:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The environment is already installed on the development machine. The commands
above reproduce it on a fresh checkout. The lock file includes exact tested
runtime and testing versions. `requirements.txt` lists direct runtime packages;
`requirements-dev.txt` adds pytest. Thread limits during fitting reduce CPU use.

## Fit clusters

```powershell
.\.venv\Scripts\python.exe -m mall_clustering train
```

The default `--k auto` selects the highest silhouette among K=2..8, breaking
ties in favour of smaller K. The elbow chart is provided for visual inspection;
no automatic elbow-detection algorithm is claimed.

Force a particular K:

```powershell
.\.venv\Scripts\python.exe -m mall_clustering train --k 5
```

Values 2-8 are supported; all candidate diagnostics are still exported when a
K is forced. Five clearly separated groups were used to create the practice
data, so recovering five groups here does not prove that real mall customers
naturally form five segments.

## Assign a new fictional customer

```powershell
.\.venv\Scripts\python.exe -m mall_clustering assign --income 60 --score 50
```

The result includes cluster ID, its centroid in original units, distance in
standardized space and fields outside the training ranges. Distance is not a
probability or confidence score. IDs may change when refitting or changing K;
compare centroids rather than assuming a numeric ID has a lasting meaning.

Default paths are relative to the project. Run module commands from this folder.
CLI-supplied relative paths follow the current shell directory:

```powershell
.\.venv\Scripts\python.exe -m mall_clustering train --data .\sample_data\customers.csv --output-dir .\artifacts
.\.venv\Scripts\python.exe -m mall_clustering assign --income 60 --score 50 --model .\artifacts\model.joblib
```

Only load model files you created or trust: joblib uses pickle, which can execute
code from an untrusted file. Use matching recorded package versions when loading.

## Outputs

| File under `artifacts/` | Contents |
| --- | --- |
| model.joblib | Fitted scaler, K-means and metadata |
| metrics.json | Selected K, descriptive metrics and candidate scores |
| clustered_customers.csv | Every input row and assigned cluster ID |
| centroids.csv | Original-unit centers and customer counts |
| k_evaluation.csv | Inertia and silhouette for candidate values |
| clusters.png | Customer groups and centroid stars in original units |
| elbow.png | Inertia versus K |
| silhouette.png | Silhouette versus K |

Inertia sums squared distances to assigned centers in standardized space and
usually decreases as K grows. It cannot be minimized alone to choose a useful K.
Silhouette compares within-cluster cohesion with separation, ranges from -1 to 1,
and requires at least two clusters. Neither metric establishes business value
or proves the existence of true customer categories. The results depend on
chosen features, scaling, sampling and the K-means preference for compact groups.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Tests cover deterministic data generation, validation, candidate selection,
scaling, independently recomputed inertia, silhouette, cluster counts and means,
saved nearest-centroid assignment, forced K, invalid input, PNG decoding and
separate-process command-line execution.

VS Code **F5** fits the clusters. **Terminal > Run Task** includes fitting,
fictional assignment and tests. No server, external account or PowerShell policy
changes are needed.

## Structure

```text
mall_clustering/
  data.py       CSV and feature validation
  model.py      Candidate fits, selection, summaries and assignment
  plots.py      Cluster, elbow and silhouette charts
  __main__.py   Command-line interface
sample_data/    Fictional CSV and its provenance
scripts/        Reproducible sample-data generator
tests/          Data, clustering and command-line checks
.vscode/        Interpreter, debugger and tasks
artifacts/      Generated outputs, ignored by Git
```

## GitHub submission

Include source, fictional dataset/generator, tests, README, requirements, pytest
and VS Code settings. Generated artifacts, environment, caches and local test
report are ignored. No demo screenshots are included. Repository: https://github.com/Htet2345/ncc-data-science-pj-10

## References

- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
- [scikit-learn StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
