"""Preprocessing for German Credit data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline.common import load_config, resolve_path
from src.pipeline.schema import validate_config_schema, validate_dataframe_features


def get_data_path(config: dict, key: str) -> Path:
    """Resolve a configured data path key."""
    return resolve_path(config["data"][key])


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load and validate raw dataset columns."""
    validate_config_schema(config)

    data_path = get_data_path(config, "raw_path")
    column_names = config["data"]["column_names"]
    target_col = config["schema"]["target_column"]
    expected_rows = int(config["data"].get("expected_rows", 1000))
    expected_feature_count = int(config["data"].get("expected_feature_count", 20))

    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None,
        names=column_names,
        engine="python",
    )
    if df.empty:
        raise ValueError("Raw dataset is empty.")
    if df.shape[1] != len(column_names):
        raise ValueError(
            f"Unexpected column count. Expected {len(column_names)}, got {df.shape[1]}."
        )
    observed_feature_count = df.shape[1] - 1
    if observed_feature_count != expected_feature_count:
        raise ValueError(
            "Unexpected feature count. "
            f"Expected {expected_feature_count}, got {observed_feature_count}."
        )
    if df.shape[0] != expected_rows:
        raise ValueError(
            f"Unexpected row count. Expected {expected_rows}, got {df.shape[0]}."
        )

    validate_dataframe_features(df, target_col)
    return df


def cast_columns(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Cast numeric columns to numeric dtype and categoricals to string."""
    target_col = config["schema"]["target_column"]
    numeric_cols = config["schema"]["numeric_columns"] + [target_col]
    categorical_cols = config["schema"]["categorical_columns"]

    casted = df.copy()
    for column in numeric_cols:
        casted[column] = pd.to_numeric(casted[column], errors="raise")
    for column in categorical_cols:
        casted[column] = casted[column].astype(str)
    return casted


def remap_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remap UCI labels to {0,1}."""
    target_col = config["schema"]["target_column"]
    mapping_cfg = config["pipeline"]["target_mapping"]
    dist_cfg = config["data"].get("expected_class_distribution", {})
    mapping = {
        mapping_cfg["original_good"]: mapping_cfg["mapped_good"],
        mapping_cfg["original_bad"]: mapping_cfg["mapped_bad"],
    }

    remapped = df.copy()
    remapped[target_col] = remapped[target_col].map(mapping)
    if remapped[target_col].isna().any():
        raise ValueError("Unexpected target values found while remapping labels.")
    remapped[target_col] = remapped[target_col].astype(int)

    total_count = remapped[target_col].shape[0]
    good_count = int((remapped[target_col] == mapping_cfg["mapped_good"]).sum())
    bad_count = int((remapped[target_col] == mapping_cfg["mapped_bad"]).sum())
    good_ratio = good_count / total_count
    bad_ratio = bad_count / total_count

    expected_good = float(dist_cfg.get("good_ratio", 0.7))
    expected_bad = float(dist_cfg.get("bad_ratio", 0.3))
    tolerance = float(dist_cfg.get("tolerance", 0.1))
    if abs(good_ratio - expected_good) > tolerance or abs(bad_ratio - expected_bad) > tolerance:
        raise ValueError(
            "Class distribution is outside expected bounds. "
            f"Observed good={good_ratio:.3f}, bad={bad_ratio:.3f}; "
            f"expected good~{expected_good}, bad~{expected_bad} (+/- {tolerance})."
        )

    print(
        "[preprocess] Target remap complete: "
        f"0=Good({good_count}, {good_ratio:.1%}), "
        f"1=Bad({bad_count}, {bad_ratio:.1%})"
    )
    return remapped


def save_processed_data(df: pd.DataFrame, config: dict) -> Path:
    """Persist processed data to CSV."""
    output_path = get_data_path(config, "processed_path")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def preprocess_data(config: dict) -> pd.DataFrame:
    """Run preprocessing end-to-end."""
    df = load_raw_data(config)
    df = cast_columns(df, config)
    df = remap_target(df, config)
    output_path = save_processed_data(df, config)
    print(f"[preprocess] Processed data saved to {output_path}")
    return df


def main() -> None:
    """CLI entrypoint."""
    config = load_config()
    df = preprocess_data(config)
    print(f"[preprocess] Completed with shape={df.shape}")


if __name__ == "__main__":
    main()
