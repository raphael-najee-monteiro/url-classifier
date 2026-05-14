"""Load latest registered model and predict."""
from __future__ import annotations
import mlflow
import pandas as pd
from src.features.url_features import extract, FEATURE_COLUMNS

_MODEL = None


def get_model(model_name: str = "url-classifier-prod", stage: str = "None"):
    global _MODEL
    if _MODEL is None:
        _MODEL = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
    return _MODEL


def predict(url: str, model_name: str = "url-classifier-prod") -> dict:
    feats = extract(url)
    X = pd.DataFrame([feats])[FEATURE_COLUMNS]
    model = get_model(model_name)
    proba = float(model.predict_proba(X)[0, 1])
    return {"url": url, "malicious_probability": proba, "label": int(proba >= 0.5), "features": feats}
