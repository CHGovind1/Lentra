"""Feature_Engineering Module - Create derived features"""
import pandas as pd
import yaml
import os
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / config_path
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe."""
    df = df.copy()
    
    # Feature 1: Monthly installment burden
    # credit_amount / duration = monthly payment
    df['monthly_payment'] = df['credit_amount'] / df['duration']
    
    # Feature 2: Debt load proxy
    # installment_commitment * credit_amount
    df['debt_load'] = df['installment_commitment'] * df['credit_amount']
    
    # Feature 3: Age brackets
    def get_age_bracket(age):
        if age < 25:
            return 'young'
        elif age < 35:
            return 'adult'
        elif age < 50:
            return 'middle_aged'
        else:
            return 'senior'
    
    df['age_bracket'] = df['age'].apply(get_age_bracket)
    
    # Feature 4: Credit amount to age ratio
    df['credit_to_age'] = df['credit_amount'] / df['age']
    
    print(f"Added 4 engineered features. New shape: {df.shape}")
    return df


def get_feature_columns() -> list:
    """Get list of feature columns for training."""
    return [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'employment', 'installment_commitment',
        'personal_status', 'other_parties', 'residence_since', 'property_magnitude',
        'age', 'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker',
        # Engineered features
        'monthly_payment', 'debt_load', 'age_bracket', 'credit_to_age'
    ]


if __name__ == "__main__":
    # Test the feature engineering
    config = load_config()
    
    # Load processed data
    data_path = Path(config['data']['processed_path'])
    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        data_path = project_root / data_path
    
    df = pd.read_csv(data_path)
    df = add_features(df)
    print(df.head())
    print(f"Features: {list(df.columns)}")
