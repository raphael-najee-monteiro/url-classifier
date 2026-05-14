"""Load model from GCS and predict."""
from __future__ import annotations
import io
import json
import os
import joblib
import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account
from src.features.url_features import extract, FEATURE_COLUMNS

_MODEL = None
GCS_BUCKET = "url-classifier-mlops"
GCS_MODEL_PATH = "models/url_classifier.joblib"


def _gcs_client() -> storage.Client:
    key_json = os.environ.get("GCP_SA_KEY")
    if key_json:
        info = json.loads(key_json)
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(credentials=creds, project=info.get("project_id"))
    return storage.Client()  # uses ADC when running locally


def get_model():
    global _MODEL
    if _MODEL is None:
        client = _gcs_client()
        buf = io.BytesIO()
        client.bucket(GCS_BUCKET).blob(GCS_MODEL_PATH).download_to_file(buf)
        buf.seek(0)
        _MODEL = joblib.load(buf)
    return _MODEL


def predict(url: str) -> dict:
    feats = extract(url)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    model = get_model()
    proba = float(model.predict_proba(X)[0, 1])
    return {"url": url, "malicious_probability": proba, "label": int(proba >= 0.5), "features": feats}
