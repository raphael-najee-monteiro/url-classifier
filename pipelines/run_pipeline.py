"""Run the full retraining pipeline: fetch -> extract -> train & register."""
from pipelines.tasks.fetch import fetch_urls
from pipelines.tasks.features import extract_features
from src.training.train import main as train_and_register

if __name__ == "__main__":
    print("=== Step 1/3: Fetch URLs ===")
    fetch_urls()

    print("\n=== Step 2/3: Extract Features ===")
    extract_features()

    print("\n=== Step 3/3: Train & Register ===")
    train_and_register()
