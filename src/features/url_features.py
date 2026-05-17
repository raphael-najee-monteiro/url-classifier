"""URL feature extraction."""
from __future__ import annotations
import math
import re
from urllib.parse import urlparse
import pandas as pd
import tldextract

SUSPICIOUS_TLDS = {"zip", "review", "country", "kim", "cricket", "science", "work", "party", "gq", "tk"}
IP_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def _normalize_url(url: str) -> str:
    """Normalize URL to a canonical form for consistent feature extraction.

    Two transformations are applied:
    1. Add an ``http://`` scheme when none is present so that urlparse can
       correctly split the netloc from the path component.
    2. Ensure the path is at least ``/`` (i.e. treat an empty path the same
       as the root path).  Without this, bare domains such as ``google.com``
       produce ``path_length=0`` while the identical URL written as
       ``google.com/`` produces ``path_length=1``, causing the model to
       classify them differently.  The training data has the same issue:
       malicious Kaggle samples are stored as bare domains (path_length=0)
       while benign samples are stored as full URLs with paths (path_length≥1),
       so the model conflates "no trailing slash" with "malicious".
    """
    if "://" not in url:
        url = f"http://{url}"
    try:
        if urlparse(url).path == "":
            url += "/"
    except ValueError:
        pass  # malformed URL (e.g. invalid IPv6); extract() will handle it
    return url


def extract(url: str) -> dict:
    """Extract features from a single URL. Returns a dict matching the feature schema."""
    url = _normalize_url(url.strip())
    try:
        parsed = urlparse(url)
    except ValueError:
        parsed = urlparse("http://invalid/")  # fallback for malformed URLs
    ext = tldextract.extract(url)
    host = parsed.hostname or ""
    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(parsed.path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "has_ip": int(bool(IP_REGEX.match(host))),
        "has_at": int("@" in url),
        "has_https": int(parsed.scheme == "https"),
        "suspicious_tld": int(ext.suffix in SUSPICIOUS_TLDS),
        "host_entropy": _entropy(host),
    }


FEATURE_COLUMNS = list(extract("http://example.com").keys())


def extract_batch(urls: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract(u) for u in urls])
