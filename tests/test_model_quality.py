"""Quality gate test — runs in CI."""
import pytest

pytestmark = pytest.mark.skip(reason="Enable once a model is registered in MLflow")


def test_registered_model_meets_threshold():
    import mlflow, yaml, pandas as pd
    from sklearn.metrics import accuracy_score
    from src.features.url_features import extract_batch, FEATURE_COLUMNS

    params = yaml.safe_load(open("params.yaml"))
    model = mlflow.sklearn.load_model(f"models:/{params['mlflow']['registered_model_name']}/latest")
    df = pd.read_csv(params["data"]["raw_path"]).sample(500, random_state=0)
    X = extract_batch(df["url"])[FEATURE_COLUMNS]
    assert accuracy_score(df["label"], model.predict(X)) >= params["quality_gate"]["min_accuracy"]
