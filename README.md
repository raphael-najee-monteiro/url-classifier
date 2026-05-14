# MLOps Project: Malicious URL Classifier

End-to-end MLOps pipeline for real-time phishing URL detection — from raw data to a live web app, with automated daily retraining.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GitHub Repository                          │
│                                                                     │
│  Every push  ──► GitHub Actions CI ──► pytest (quality gate)        │
│                                                                     │
│  Daily cron  ──► GitHub Actions retrain pipeline                    │
│                    │                                                │
│                    ├── fetch_urls    (PhishTank + Tranco + Kaggle)  │
│                    ├── extract_features  (12 URL features → parquet)│
│                    └── train + register ──► MLflow on Cloud Run     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │        Google Cloud         │
              │                             │
              │  Cloud Storage (GCS)        │
              │  ├── urls.csv               │
              │  └── model.joblib           │
              │                             │
              │  Cloud Run                  │
              │  └── MLflow Tracking Server │
              └──────────────┬──────────────┘
                             │  downloads model.joblib on startup
              ┌──────────────▼──────────────┐
              │   Streamlit Community Cloud │
              │   URL Shield — live web app │
              └─────────────────────────────┘
```

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Feature engineering | scikit-learn + custom | 12 handcrafted URL features |
| Model | GradientBoostingClassifier | Phishing classification |
| Experiment tracking | **MLflow** on Cloud Run | Logs params, metrics, models |
| Dataset versioning | **DVC** | Tracks `urls.csv` and `features.parquet` in GCS |
| Retraining | **GitHub Actions** cron | Daily fetch → extract → train |
| Data sources | PhishTank + Tranco + Kaggle | Fresh phishing + benign URLs |
| Model storage | **GCS** bucket | `model.joblib` artifact |
| Data storage | **GCS** bucket | `urls.csv`, `features.parquet` |
| MLflow backend | **Cloud Run** (Docker) | Persistent experiment UI |
| Web app | **Streamlit Community Cloud** | Free hosted prediction UI |
| CI | **GitHub Actions** + pytest | Test on every push / PR |

---

## Project Structure

```
url-classifier/
├── app/
│   └── streamlit_app.py          # Streamlit web UI (URL Shield)
├── data/
│   ├── raw/kaggle.csv            # Kaggle phishing dataset (~822k URLs, source)
│   ├── raw/urls.csv              # Canonical training file (generated, tracked in DVC)
│   └── processed/features.parquet  # Extracted features (generated)
├── mlflow-gcp/
│   └── Dockerfile                # MLflow tracking server for Cloud Run
├── pipelines/
│   ├── run_pipeline.py           # Manual runner: fetch → extract → train
│   ├── tasks/
│   │   ├── fetch.py              # fetch_urls: Kaggle bootstrap + PhishTank + Tranco
│   │   └── features.py           # extract_features: URL → 12 features → parquet
│   └── dags/
│       └── retrain_dag.py        # Airflow DAG definition (daily schedule)
├── src/
│   ├── features/url_features.py  # Feature extraction logic (12 features)
│   ├── models/baseline.py        # GradientBoostingClassifier builder
│   ├── training/train.py         # Train + log to MLflow + quality gate
│   └── inference/predict.py      # Load model from MLflow + predict
├── tests/
│   ├── test_features.py          # Unit tests for feature extraction
│   └── test_model_quality.py     # CI quality gate (accuracy ≥ 0.90, F1 ≥ 0.88)
├── .github/workflows/ci.yml      # GitHub Actions: test on push, retrain on schedule
├── dvc.yaml                      # DVC pipeline stage (train)
├── params.yaml                   # Central config: paths, hyperparams, thresholds
└── requirements.txt
```

---

## Data Pipeline

### Sources

| Source | Type | Volume | Update frequency |
|---|---|---|---|
| [Kaggle phishing dataset](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls) | Phishing + benign | ~822k URLs | One-time bootstrap |
| [PhishTank](https://www.phishtank.com) | Phishing (verified) | 5 000 / run | Daily |
| [Tranco top-1M](https://tranco-list.eu) | Benign (popular domains) | 5 000 / run | Daily |

### Flow

```
kaggle.csv (one-time)
    └──► fetch_urls() ──► urls.csv  ◄── daily: PhishTank + Tranco appended
                               │
                        extract_features()
                               │
                        features.parquet  (12 features + label per URL)
                               │
                           train()  ──► MLflow run logged
                               │           └── model registered if quality gate passes
                           model.joblib ──► GCS bucket
```

### Label convention

Kaggle ships with `status=1` for legitimate URLs. We invert to match our convention:

```
label = 1 - status
  → label=0  benign
  → label=1  malicious (phishing)
```

---

## Features Extracted per URL (12 total)

| Feature | Description |
|---|---|
| `url_length` | Total character count |
| `host_length` | Hostname character count |
| `path_length` | URL path character count |
| `num_dots` | Number of dots (subdomain depth signal) |
| `num_hyphens` | Number of hyphens (common in phishing domains) |
| `num_digits` | Number of digit characters |
| `num_subdomains` | Subdomain depth |
| `has_ip` | 1 if hostname is a raw IP address |
| `has_at` | 1 if URL contains `@` (credential-stuffing trick) |
| `has_https` | 1 if scheme is `https` |
| `suspicious_tld` | 1 if TLD is in a known-bad list (`.tk`, `.gq`, etc.) |
| `host_entropy` | Shannon entropy of the hostname (randomness signal) |

---

## Model & Quality Gate

**Model:** `GradientBoostingClassifier` (scikit-learn)

Hyperparameters (in `params.yaml`):

```yaml
model:
  n_estimators: 50
  max_depth: 5
  learning_rate: 0.1
```

**Quality gate** — model is only registered in MLflow if both thresholds pass:

```yaml
quality_gate:
  min_accuracy: 0.90
  min_f1: 0.88
```

If the gate fails, the run is logged but not registered and the previous production model remains active.

---

## MLflow Tracking Server (Cloud Run)

The MLflow server runs as a Docker container on **Google Cloud Run** with a GCS backend store.

**Dockerfile** (`mlflow-gcp/Dockerfile`):

```dockerfile
FROM ghcr.io/mlflow/mlflow:v3.10.0-full
RUN pip install google-cloud-storage
```

**Deploy command:**

```bash
gcloud run deploy mlflow-gcp \
  --source mlflow-gcp/ \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars MLFLOW_BACKEND_STORE_URI=gs://url-classifier-mlops/mlflow \
  --set-env-vars MLFLOW_ARTIFACTS_DESTINATION=gs://url-classifier-mlops/mlflow-artifacts
```

The tracking URI is set in `params.yaml`:

```yaml
mlflow:
  tracking_uri: https://mlflow-gcp-528928377097.europe-west1.run.app
  experiment_name: malicious-url-classifier
  registered_model_name: url-classifier-prod
```

---

## GCP Setup (one-time)

### Prerequisites

- GCP project with billing enabled
- [`gcloud` CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated

### 1. Enable APIs

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud services enable storage.googleapis.com run.googleapis.com artifactregistry.googleapis.com
```

### 2. Create the GCS bucket

```bash
gsutil mb -l europe-west6 gs://url-classifier-mlops
```

### 3. Create a service account

```bash
gcloud iam service-accounts create url-classifier-sa \
  --display-name "URL Classifier SA"

PROJECT_ID=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:url-classifier-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:url-classifier-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:url-classifier-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### 4. Download the JSON key

```bash
gcloud iam service-accounts keys create ~/url-classifier-key.json \
  --iam-account="url-classifier-sa@${PROJECT_ID}.iam.gserviceaccount.com"
```

### 5. Add the key as secrets

| Location | Secret name | Value |
|---|---|---|
| GitHub → Settings → Secrets | `GCP_SA_KEY` | Full contents of `url-classifier-key.json` |
| Streamlit Community Cloud → App secrets | `GCP_SA_KEY` | Full contents of `url-classifier-key.json` |

---

## Local Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Pull dataset via DVC (recommended)

```bash
gcloud auth login
gcloud auth application-default login
dvc pull
```

### Or download the Kaggle dataset manually

1. Go to [Kaggle phishing dataset](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls)
2. Download → extract zip → rename to `kaggle.csv`
3. Place at `data/raw/kaggle.csv`

---

## Running the Pipeline

### All steps at once

```bash
python -m pipelines.run_pipeline
```

### Or step by step

```bash
# 1. Prepare training data
#    First run: normalizes kaggle.csv into urls.csv
#    Subsequent runs: appends fresh PhishTank + Tranco URLs
python -m pipelines.tasks.fetch

# 2. Extract features → data/processed/features.parquet
python -m pipelines.tasks.features

# 3. Train, evaluate, and register model (if quality gate passes)
python -m src.training.train

# 4. Launch the web app  →  http://localhost:8501
streamlit run app/streamlit_app.py

# 5. Inspect MLflow experiments (optional)  →  http://localhost:5000
mlflow ui
```

---

## Automated Retraining (GitHub Actions)

The retraining workflow runs **daily at 02:00 UTC** and mirrors the manual pipeline:

```
fetch_urls → extract_features → train_and_register
```

It authenticates to GCP using the `GCP_SA_KEY` secret, reads `urls.csv` from GCS, trains the model, and uploads the new `model.joblib` back to GCS. If the quality gate passes, the model is also registered in the MLflow registry on Cloud Run.

The Airflow DAG (`pipelines/dags/retrain_dag.py`) defines the same pipeline for self-hosted Airflow deployments.

---

## CI — GitHub Actions

On every push and pull request:

```
checkout → install dependencies → pytest -v
```

Tests include:
- **`test_features.py`** — unit tests for all 12 feature extractors
- **`test_model_quality.py`** — checks that the registered model meets the quality gate thresholds (skipped until a model is registered)

---

## Streamlit App — URL Shield

Live at Streamlit Community Cloud. On startup, the app downloads the latest `model.joblib` from GCS using the service account key stored in Streamlit secrets.

Enter any URL to get:
- **Verdict** — Likely Safe / Likely Malicious
- **Risk score** — malicious probability (0–100%)
- **Feature breakdown** — all 12 extracted feature values

---

## Configuration Reference (`params.yaml`)

```yaml
data:
  raw_path: data/raw/urls.csv
  processed_path: data/processed/features.parquet
  test_size: 0.2
  random_state: 42

fetch:
  kaggle_path: data/raw/kaggle.csv
  phishtank_n: 5000       # phishing URLs fetched per daily run
  tranco_n: 5000          # benign URLs fetched per daily run

model:
  n_estimators: 50
  max_depth: 5
  learning_rate: 0.1

mlflow:
  experiment_name: malicious-url-classifier
  registered_model_name: url-classifier-prod
  tracking_uri: https://mlflow-gcp-528928377097.europe-west1.run.app

quality_gate:
  min_accuracy: 0.90
  min_f1: 0.88
```
