"""Model Evaluation Module - Compute and log metrics"""
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_model(model, X_val, y_val) -> dict:
    """Compute evaluation metrics."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    metrics = {
        "auc_roc": roc_auc_score(y_val, y_prob),
        "f1": f1_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
    }
    
    return metrics


def print_evaluation(metrics: dict):
    """Print evaluation metrics in a readable format."""
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.upper():15s}: {value:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Evaluation module - import this in train.py to use")
