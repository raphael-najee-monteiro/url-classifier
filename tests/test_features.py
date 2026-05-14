from src.features.url_features import extract, FEATURE_COLUMNS


def test_extract_returns_all_columns():
    feats = extract("http://example.com/path")
    assert set(feats.keys()) == set(FEATURE_COLUMNS)


def test_ip_url_detected():
    assert extract("http://192.168.1.1/login")["has_ip"] == 1


def test_https_detected():
    assert extract("https://example.com")["has_https"] == 1


def test_suspicious_tld():
    assert extract("http://free-money.tk")["suspicious_tld"] == 1
