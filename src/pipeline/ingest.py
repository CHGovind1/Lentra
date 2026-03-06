"""Data Ingestion Module - Downloads german.data from UCI"""
import urllib.request
from pathlib import Path
import yaml
import os


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    # Get absolute path to config
    if not os.path.isabs(config_path):
        # Get project root (parent of src directory = LentraTask)
        # src/pipeline/ingest.py -> parent.parent.parent = LentraTask
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        config_path = project_root / config_path
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_data_path(config: dict) -> Path:
    """Get absolute path to data file."""
    raw_path = config['data']['raw_path']
    path = Path(raw_path)
    
    # If relative path, make it absolute from project root
    if not path.is_absolute():
        current = Path(__file__).resolve()
        project_root = current.parent.parent.parent
        path = project_root / raw_path
    
    return path


def download_data(config: dict) -> str:
    """Download data from configured URL."""
    data_url = config['data']['url']
    data_path = get_data_path(config)
    
    # Create directory
    data_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading data from: {data_url}")
    print(f"Saving to: {data_path}")
    
    urllib.request.urlretrieve(data_url, data_path)
    print(f"Data downloaded successfully!")
    return str(data_path)


def load_data(config: dict) -> str:
    """Load data - download if not exists."""
    data_path = get_data_path(config)
    
    if data_path.exists():
        print(f"Data already exists at: {data_path}")
        return str(data_path)
    else:
        return download_data(config)


if __name__ == "__main__":
    config = load_config()
    data_path = load_data(config)
    print(f"Data ready at: {data_path}")
