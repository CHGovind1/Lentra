"""Feature engineering shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_age_group_bins(config: dict) -> tuple[list[float], list[str], bool]:
    age_cfg = config.get("features", {}).get("age_group", {})
    raw_bins = age_cfg.get("bins", [0, 25, 35, 50, None])
    bins: list[float] = []
    for value in raw_bins:
        bins.append(np.inf if value is None else float(value))

    labels = age_cfg.get("labels", ["young", "adult", "mid_age", "senior"])
    right_inclusive = bool(age_cfg.get("right_inclusive", False))
    return bins, labels, right_inclusive


def add_engineered_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create configured derived features from base German Credit fields."""
    features = df.copy()

    numeric_features = config["features"]["engineered_numeric_columns"]
    categorical_features = config["features"]["engineered_categorical_columns"]

    duration = features["duration"].clip(lower=1)
    existing_credits = features["existing_credits"].clip(lower=1)

    for feature_name in numeric_features:
        if feature_name == "monthly_payment":
            features["monthly_payment"] = features["credit_amount"] / duration
        elif feature_name == "debt_burden":
            features["debt_burden"] = (
                features["installment_commitment"] * features["credit_amount"]
            )
        elif feature_name == "credit_per_existing_credit":
            features["credit_per_existing_credit"] = (
                features["credit_amount"] / existing_credits
            )
        else:
            raise ValueError(f"Unsupported engineered numeric feature: {feature_name}")

    for feature_name in categorical_features:
        if feature_name == "age_group":
            bins, labels, right_inclusive = _resolve_age_group_bins(config)
            age_group = pd.cut(
                features["age"],
                bins=bins,
                labels=labels,
                right=right_inclusive,
                include_lowest=True,
            )
            features["age_group"] = (
                age_group.astype("string").fillna("unknown").astype(str)
            )
        else:
            raise ValueError(
                f"Unsupported engineered categorical feature: {feature_name}"
            )

    return features
