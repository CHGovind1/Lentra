"""Model Training Module - Train and log to MLflow"""
import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """Load processed data."""
    data_path = Path(config['data']['processed_path'])
    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        data_path = project_root / data_path
    
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple:
    """Encode categorical features."""
    df = df.copy()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Store encoders for later use
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    print(f"Encoded {len(categorical_cols)} categorical columns")
    return df, encoders


def prepare_data(df: pd.DataFrame, config: dict) -> tuple:
    """Prepare features and target for training."""
    # Add features
    from src.pipeline.features import add_features
    df = add_features(df)
    
    # Encode categoricals
    df, encoders = encode_categoricals(df)
    
    # Get feature columns
    feature_cols = [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment', 'installment_commitment',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker',
        'monthly_payment', 'debt_load', 'credit_to_age'
    ]
    
    # Handle age_bracket if it exists
    if 'age_bracket' in df.columns:
        feature_cols.append('age_bracket')
    
    X = df[feature_cols]
    y = df['class']
    
    # Split data
    train_ratio = config['pipeline']['train_split']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=1-train_ratio,
        random_state=config['pipeline']['random_seed']
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    return X_train, X_val, y_train, y_val, encoders


def train_model(X_train, y_train, config: dict):
    """Train the model based on config."""
    model_type = config['model']['type']
    params = config['model']['params']
    
    if model_type == 'xgboost':
        model = xgb.XGBClassifier(**params)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**params)
    else:
        model = RandomForestClassifier(**params)
    
    model.fit(X_train, y_train)
    print(f"Trained {model_type} model")
    return model


def log_to_mlflow(model, X_val, y_val, config: dict):
    """Log model and metrics to MLflow."""
    # Set tracking URI
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', config['mlflow']['tracking_uri'])
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    experiment_name = config['mlflow']['experiment_name']
    mlflow.set_experiment(experiment_name)
    
    # Start run
    with mlflow.start_run(run_name=f"{config['mlflow']['run_name_prefix']}-1"):
        # Log parameters
        mlflow.log_params(config['model']['params'])
        mlflow.log_param("model_type", config['model']['type'])
        mlflow.log_param("train_split", config['pipeline']['train_split'])
        mlflow.log_param("random_seed", config['pipeline']['random_seed'])
        
        # Evaluate
        from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        # Log metrics
        mlflow.log_metrics({
            "auc_roc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        })
        
        # Log model
        mlflow.sklearn.log_model(
            model,
            artifact_path=config['minio']['artifacts_path'],
            registered_model_name=config['mlflow']['model_name']
        )
        
        print(f"Logged to MLflow: AUC={auc:.4f}, F1={f1:.4f}")
    
    return {"auc_roc": auc, "f1": f1, "precision": precision, "recall": recall}


if __name__ == "__main__":
    config = load_config()
    
    # Load and prepare data
    df = load_data(config)
    X_train, X_val, y_train, y_val, encoders = prepare_data(df, config)
    
    # Train model
    model = train_model(X_train, y_train, config)
    
    # Log to MLflow
    metrics = log_to_mlflow(model, X_val, y_val, config)
    
    print("Training complete!")
    print(f"Metrics: {metrics}")
