"""Preprocessing_Module - Parse, encode, and remap target"""
import pandas as pd
from pathlib import Path
import yaml
import os


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / config_path
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_path(config: dict, path_key: str = 'raw_path') -> Path:
    """Get absolute path to data file."""
    rel_path = config['data'][path_key]
    path = Path(rel_path)
    
    if not path.is_absolute():
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        path = project_root / rel_path
    
    return path


def load_raw_data(config: dict) -> pd.DataFrame:
    """Load raw data from file."""
    data_path = get_data_path(config, 'raw_path')
    column_names = config['data']['column_names']
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def remap_target(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Remap target: 1 (Good) -> 0, 2 (Bad) -> 1"""
    original_good = config['pipeline']['target_mapping']['original_good']
    original_bad = config['pipeline']['target_mapping']['original_bad']
    
    # Remap: original good (1) -> 0, original bad (2) -> 1
    df['class'] = df['class'].map({original_good: 0, original_bad: 1})
    
    print(f"Target remapped: 0 (Good)={sum(df['class']==0)}, 1 (Bad)={sum(df['class']==1)}")
    return df


def save_processed_data(df: pd.DataFrame, config: dict) -> str:
    """Save processed data to file."""
    output_path = get_data_path(config, 'processed_path')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    return str(output_path)


def preprocess(config: dict) -> pd.DataFrame:
    """Main preprocessing pipeline."""
    # Load raw data
    df = load_raw_data(config)
    
    # Remap target
    df = remap_target(df, config)
    
    # Save processed data
    save_processed_data(df, config)
    
    return df


if __name__ == "__main__":
    config = load_config()
    df = preprocess(config)
    print(f"Preprocessing complete! Shape: {df.shape}")
