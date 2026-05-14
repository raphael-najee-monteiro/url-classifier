"""Train + log to MLflow + register if quality gate passes."""
from __future__ import annotations
import io
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from google.cloud import storage
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.features.url_features import extract_batch, FEATURE_COLUMNS
from src.models.baseline import build_model

def load_params() -> dict:
    return yaml.safe_load(Path("params.yaml").read_text())

def main() -> None:
    params = load_params()
    df = pd.read_csv(params["data"]["raw_path"])  # expects columns: url, label (0/1)

    X = extract_batch(df["url"])[FEATURE_COLUMNS]
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], random_state=params["data"]["random_state"], stratify=y
    )

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params(params["model"])
        model = build_model(params["model"])
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        print(metrics)

        gate = params["quality_gate"]
        if metrics["accuracy"] >= gate["min_accuracy"] and metrics["f1"] >= gate["min_f1"]:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=params["mlflow"]["registered_model_name"],
            )
            print("Model registered.")

            buf = io.BytesIO()
            joblib.dump(model, buf)
            buf.seek(0)
            gcs = storage.Client()
            gcs.bucket("url-classifier-mlops").blob("models/url_classifier.joblib").upload_from_file(buf)
            print("Model uploaded to gs://url-classifier-mlops/models/url_classifier.joblib")
        else:
            print("Quality gate failed — not registered.")


if __name__ == "__main__":
    main()
