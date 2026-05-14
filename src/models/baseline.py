"""Baseline model."""
from sklearn.ensemble import GradientBoostingClassifier


def build_model(params: dict) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=params.get("n_estimators", 200),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.1),
        random_state=42,
    )
