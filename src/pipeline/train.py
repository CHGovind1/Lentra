"""Training  entrypoint for the German Credit classification pipeline."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import urllib.error
import urllib.request

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.pipeline.common import load_config, resolve_path, utc_now_iso
from src.pipeline.evaluate import compute_metrics, print_metrics
from src.pipeline.features import add_engineered_features
from src.pipeline.ingest import ensure_data_available
from src.pipeline.preprocess import preprocess_data


REQUIRED_EVAL_METRICS = ("auc_roc", "f1", "precision", "recall")


def set_random_seed(seed: int) -> None:
    """Set deterministic seeds used by this pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_classifier(config: dict[str, Any]):
    """Create classifier instance from config."""
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    params = dict(model_cfg.get("params", {}))

    if model_type == "logistic_regression":
        return LogisticRegression(**params)
    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    raise ValueError(f"Unsupported model type: {model_type}")


def build_training_pipeline(config: dict[str, Any]) -> Pipeline:
    """Create sklearn pipeline for features, preprocessing, and training."""
    schema = config["schema"]
    feature_cfg = config["features"]
    categorical_cfg = config.get("preprocessing", {}).get("categorical_handling", {})
    categorical_strategy = categorical_cfg.get("strategy", "opaque_one_hot")
    if categorical_strategy != "opaque_one_hot":
        raise ValueError(
            "Unsupported categorical handling strategy: "
            f"{categorical_strategy}. Only 'opaque_one_hot' is supported."
        )

    numeric_cols = list(schema["numeric_columns"]) + list(
        feature_cfg["engineered_numeric_columns"]
    )
    categorical_cols = list(schema["categorical_columns"]) + list(
        feature_cfg["engineered_categorical_columns"]
    )

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Backward compatibility for older sklearn versions.
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", one_hot_encoder, categorical_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                FunctionTransformer(
                    add_engineered_features,
                    kw_args={"config": config},
                    validate=False,
                ),
            ),
            ("preprocessor", preprocessor),
            ("classifier", build_classifier(config)),
        ]
    )


def split_dataset(
    df: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split processed dataset into train/validation sets."""
    target_col = config["schema"]["target_column"]
    test_size = float(config["pipeline"]["test_size"])
    seed = int(config["pipeline"]["random_seed"])

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def flatten_params(prefix: str, params: dict[str, Any]) -> dict[str, Any]:
    """Flatten model params for MLflow logging."""
    return {f"{prefix}{key}": value for key, value in params.items()}


def flatten_for_mlflow(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested config into mlflow-compatible param key/value pairs."""
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}{key}."
            flattened.update(flatten_for_mlflow(nested_prefix, nested_value))
        return flattened
    if isinstance(value, (list, tuple)):
        flattened[prefix[:-1]] = json.dumps(value)
        return flattened
    flattened[prefix[:-1]] = value
    return flattened


def tracking_uri_reachable(tracking_uri: str) -> bool:
    """Check whether an HTTP(S) tracking URI is reachable."""
    parsed = urlparse(tracking_uri)
    if parsed.scheme not in {"http", "https"}:
        return True
    try:
        with urllib.request.urlopen(tracking_uri, timeout=3):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def validate_metrics_for_tracking(metrics: dict[str, Any]) -> None:
    """Ensure required evaluation metrics are present for MLflow logging."""
    missing = [name for name in REQUIRED_EVAL_METRICS if name not in metrics]
    if missing:
        raise ValueError(
            "Missing required evaluation metrics for MLflow logging: "
            + ", ".join(missing)
        )


def wait_for_tracking_uri(tracking_uri: str, timeout_seconds: int = 30) -> bool:
    """Wait briefly for MLflow tracking URI readiness."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if tracking_uri_reachable(tracking_uri):
            return True
        time.sleep(1)
    return False


def get_git_commit_hash() -> str | None:
    """Return current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            cwd=resolve_path("."),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def ensure_writable_temp_dir(config: dict[str, Any]) -> Path:
    """Force temp files to a writable workspace directory."""
    artifacts_dir = resolve_path(config["artifacts"]["base_dir"])
    temp_dir = artifacts_dir / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temp_dir)
    os.environ["TMP"] = str(temp_dir)
    os.environ["TEMP"] = str(temp_dir)
    return temp_dir


def save_local_artifacts(
    model: Pipeline,
    metrics: dict[str, Any],
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    """Persist local model, metrics, and metadata artifacts."""
    artifacts_cfg = config["artifacts"]
    artifacts_dir = resolve_path(artifacts_cfg["base_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / artifacts_cfg["model_filename"]
    metrics_path = artifacts_dir / artifacts_cfg["metrics_filename"]
    metadata_path = artifacts_dir / artifacts_cfg["metadata_filename"]

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "artifacts_dir": artifacts_dir,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "metadata_path": metadata_path,
    }


def log_to_mlflow(
    model: Pipeline,
    X_train: pd.DataFrame,
    metrics: dict[str, Any],
    config: dict[str, Any],
    artifact_paths: dict[str, Path],
) -> dict[str, Any] | None:
    """Log run metadata, metrics, and model to MLflow."""
    mlflow_cfg = config["mlflow"]
    enabled = bool(mlflow_cfg.get("enabled", True))
    required = bool(mlflow_cfg.get("required", True))

    if not enabled:
        if required:
            raise RuntimeError("MLflow is required but config.mlflow.enabled=false.")
        print("[train] MLflow logging disabled by config.")
        return None

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI",
        mlflow_cfg["tracking_uri"],
    )
    tracking_ready = wait_for_tracking_uri(tracking_uri, timeout_seconds=45)
    if not tracking_ready:
        message = f"MLflow tracking URI unreachable: {tracking_uri}"
        if required:
            raise RuntimeError(message)
        print(f"[train] {message}. Skipping MLflow logging.")
        return None

    validate_metrics_for_tracking(metrics)
    if required and not mlflow_cfg.get("register_model", False):
        raise RuntimeError(
            "MLflow is required but config.mlflow.register_model=false. "
            "Set it to true to register model artifacts."
        )

    mlflow.set_tracking_uri(tracking_uri)
    ensure_writable_temp_dir(config)

    experiment_name = mlflow_cfg["experiment_name"]
    run_name = (
        f"{mlflow_cfg['run_name_prefix']}-"
        f"{config['model']['type']}-"
        f"{utc_now_iso()}"
    )
    model_params = config["model"].get("params", {})
    run_tags = dict(mlflow_cfg.get("run_tags", {}))
    run_tags.update(
        {
            "model_type": config["model"]["type"],
            "categorical_strategy": config.get("preprocessing", {})
            .get("categorical_handling", {})
            .get("strategy", "opaque_one_hot"),
        }
    )
    git_commit = get_git_commit_hash()
    if git_commit:
        run_tags["git_commit"] = git_commit

    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.set_tags(run_tags)

            mlflow.log_param("model_type", config["model"]["type"])
            mlflow.log_param("random_seed", config["pipeline"]["random_seed"])
            mlflow.log_param("test_size", config["pipeline"]["test_size"])
            mlflow.log_param(
                "categorical_strategy",
                config.get("preprocessing", {})
                .get("categorical_handling", {})
                .get("strategy", "opaque_one_hot"),
            )
            mlflow.log_params(flatten_params("model__", model_params))
            mlflow.log_params(
                flatten_for_mlflow("pipeline.", config.get("pipeline", {}))
            )
            mlflow.log_params(
                flatten_for_mlflow("features.", config.get("features", {}))
            )
            mlflow.log_params(
                flatten_for_mlflow(
                    "preprocessing.",
                    config.get("preprocessing", {}),
                )
            )
            mlflow.log_metrics({metric: float(metrics[metric]) for metric in REQUIRED_EVAL_METRICS})

            sample = X_train.head(5)
            signature = infer_signature(sample, model.predict(sample))

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=mlflow_cfg["artifact_path"],
                input_example=sample,
                signature=signature,
                registered_model_name=mlflow_cfg["model_name"],
            )

            model_version = getattr(model_info, "registered_model_version", None)
            target_stage = mlflow_cfg.get("registered_model_stage")
            if target_stage and model_version:
                client = MlflowClient(tracking_uri=tracking_uri)
                client.transition_model_version_stage(
                    name=mlflow_cfg["model_name"],
                    version=model_version,
                    stage=target_stage,
                    archive_existing_versions=True,
                )

            mlflow.log_artifact(str(artifact_paths["metrics_path"]), artifact_path="reports")
            mlflow.log_artifact(str(artifact_paths["metadata_path"]), artifact_path="reports")

            return {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "model_uri": model_info.model_uri,
                "model_version": model_version,
                "model_stage": target_stage,
                "tracking_uri": tracking_uri,
            }
    except Exception as exc:
        if required:
            raise RuntimeError(f"MLflow logging failed: {exc}") from exc
        print(f"[train] MLflow logging skipped due to error: {exc}")
        return None


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the full training pipeline."""
    seed = int(config["pipeline"]["random_seed"])
    set_random_seed(seed)

    ensure_data_available(config)
    processed_df = preprocess_data(config)

    X_train, X_valid, y_train, y_valid = split_dataset(processed_df, config)
    pipeline = build_training_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    y_prob = pipeline.predict_proba(X_valid)[:, 1] if hasattr(pipeline, "predict_proba") else None
    metrics = compute_metrics(y_valid.to_numpy(), y_pred, y_prob)
    print_metrics(metrics)

    metadata: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "git_commit": get_git_commit_hash(),
        "dataset_rows": int(processed_df.shape[0]),
        "dataset_columns": int(processed_df.shape[1]),
        "train_rows": int(X_train.shape[0]),
        "validation_rows": int(X_valid.shape[0]),
        "target_column": config["schema"]["target_column"],
        "model_type": config["model"]["type"],
        "categorical_handling": config.get("preprocessing", {}).get(
            "categorical_handling",
            {"strategy": "opaque_one_hot"},
        ),
    }

    artifact_paths = save_local_artifacts(pipeline, metrics, config, metadata)
    mlflow_info = log_to_mlflow(pipeline, X_train, metrics, config, artifact_paths)
    if mlflow_info:
        metadata["mlflow"] = mlflow_info
        artifact_paths["metadata_path"].write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    print(f"[train] Model artifact: {artifact_paths['model_path']}")
    print(f"[train] Metrics artifact: {artifact_paths['metrics_path']}")
    print(f"[train] Metadata artifact: {artifact_paths['metadata_path']}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train German Credit classifier")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
