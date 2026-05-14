"""Fetch and prepare URL training data."""
from __future__ import annotations
import io
import zipfile
import requests
import pandas as pd
import yaml
from pathlib import Path

PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.csv"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"


def _load_params() -> dict:
    return yaml.safe_load(Path("params.yaml").read_text())


def _normalize_kaggle(kaggle_path: str, out_path: Path) -> None:
    """Convert kaggle.csv schema → urls.csv schema.

    Kaggle uses status=1 for legitimate, status=0 for phishing.
    Our convention is label=0 for benign, label=1 for malicious.
    So we invert: label = 1 - status.
    """
    df = pd.read_csv(kaggle_path)
    df = df.rename(columns={"status": "label"})
    df["label"] = 1 - df["label"].astype(int)
    df = df[["url", "label"]].dropna()
    df = df[df["url"].apply(lambda u: isinstance(u, str) and u.isascii() and len(u) > 3)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Bootstrapped {len(df)} rows from Kaggle -> {out_path}")


def _fetch_phishtank(n: int) -> pd.DataFrame:
    """Download the PhishTank public verified-phishing CSV dump (no API key needed).

    PhishTank publishes a daily CSV of verified phishing URLs at a public endpoint.
    We take the first n rows and label them 1 (malicious).
    """
    resp = requests.get(
        PHISHTANK_URL,
        timeout=60,
        headers={"User-Agent": "research/url-classifier-mlops"},
    )
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # PhishTank CSV columns include: phish_id, url, phish_detail_url, ...
    urls = df["url"].dropna().head(n).reset_index(drop=True)
    return pd.DataFrame({"url": urls, "label": 1})


def _fetch_tranco(n: int) -> pd.DataFrame:
    """Download top-N domains from the Tranco top-1M list as benign samples.

    Tranco (https://tranco-list.eu) is a research-grade list of popular domains
    aggregating Alexa, Majestic, Cisco Umbrella, and others. Top domains are
    overwhelmingly legitimate, making them a reliable benign signal.
    The list is published as a zip containing a CSV with columns: rank, domain.
    We prepend https:// to turn domains into full URLs and label them 0 (benign).
    """
    resp = requests.get(TRANCO_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])
    domains = df["domain"].dropna().head(n)
    urls = ("https://" + domains).reset_index(drop=True)
    return pd.DataFrame({"url": urls, "label": 0})


def fetch_urls() -> None:
    """Airflow task: bootstrap from Kaggle on first run, then daily-append fresh data.

    Flow:
    1. If urls.csv does not exist yet, normalize kaggle.csv into it (one-time bootstrap).
    2. Fetch fresh phishing URLs from PhishTank.
    3. Fetch fresh benign URLs from Tranco.
    4. Drop any URL already present in urls.csv (deduplication by exact URL string).
    5. Append the survivors and overwrite urls.csv.
    """
    params = _load_params()
    raw_path = Path(params["data"]["raw_path"])
    fetch_cfg = params["fetch"]

    # --- Bootstrap ---
    if not raw_path.exists():
        _normalize_kaggle(fetch_cfg["kaggle_path"], raw_path)

    existing = pd.read_csv(raw_path)
    seen = set(existing["url"].str.strip())

    # --- Fetch fresh data ---
    frames = []
    print("Fetching PhishTank...")
    try:
        frames.append(_fetch_phishtank(fetch_cfg["phishtank_n"]))
    except Exception as e:
        print(f"WARNING: PhishTank fetch failed ({e}), skipping.")

    print("Fetching Tranco...")
    try:
        frames.append(_fetch_tranco(fetch_cfg["tranco_n"]))
    except Exception as e:
        print(f"WARNING: Tranco fetch failed ({e}), skipping.")

    if not frames:
        print("No fresh data fetched. Training on existing dataset.")
        return

    # --- Deduplicate and append ---
    new_data = pd.concat(frames, ignore_index=True)
    new_data = new_data[~new_data["url"].str.strip().isin(seen)]

    combined = pd.concat([existing, new_data], ignore_index=True)
    combined.to_csv(raw_path, index=False)
    print(f"Added {len(new_data)} new URLs. Total in dataset: {len(combined)}")


if __name__ == "__main__":
    fetch_urls()
