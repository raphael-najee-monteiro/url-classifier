"""Extract and persist features for the training pipeline."""
from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path

from src.features.url_features import extract_batch, FEATURE_COLUMNS


def _load_params() -> dict:
    return yaml.safe_load(Path("params.yaml").read_text())


def extract_features() -> None:
    """Airflow task: extract URL features and save to parquet.

    Reads urls.csv (produced by fetch_urls), runs feature extraction on every URL,
    then writes features.parquet to data/processed/.

    Why parquet and not CSV?
    - Parquet is columnar and compressed — much faster to read for training.
    - It preserves dtypes (int vs float), so no re-casting needed later.
    - It's the standard format for ML feature stores.

    The label column is carried alongside the features so the parquet file
    is a self-contained, ready-to-train dataset.
    """
    params = _load_params()
    raw_path = Path(params["data"]["raw_path"])
    processed_path = Path(params["data"]["processed_path"])

    df = pd.read_csv(raw_path)
    print(f"Extracting features for {len(df)} URLs...")

    features = extract_batch(df["url"])[FEATURE_COLUMNS]
    features["label"] = df["label"].values

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(processed_path, index=False)
    print(f"Saved {len(features)} feature rows -> {processed_path}")


if __name__ == "__main__":
    extract_features()
