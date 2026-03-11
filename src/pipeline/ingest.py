"""Data ingestion utilities for the German Credit dataset."""
from __future__ import annotations

from pathlib import Path
import urllib.error
import urllib.request

from src.pipeline.common import load_config, resolve_path


def get_raw_data_path(config: dict) -> Path:
    """Return the resolved path of the raw dataset file."""
    return resolve_path(config["data"]["raw_path"])


def download_data(config: dict, force: bool = False) -> Path:
    """Download the dataset from UCI into the configured raw path."""
    data_url = config["data"]["url"]
    timeout = int(config["data"].get("download_timeout_seconds", 30))
    data_path = get_raw_data_path(config)

    if data_path.exists() and not force:
        print(f"[ingest] Using existing dataset at {data_path}")
        return data_path

    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ingest] Downloading dataset from {data_url}")
    with urllib.request.urlopen(data_url, timeout=timeout) as response:
        data_path.write_bytes(response.read())

    print(f"[ingest] Dataset saved to {data_path}")
    return data_path


def ensure_data_available(config: dict, force_download: bool = False) -> Path:
    """Guarantee that the raw dataset is available locally."""
    data_path = get_raw_data_path(config)
    try:
        return download_data(config, force=force_download)
    except (urllib.error.URLError, TimeoutError) as exc:
        if data_path.exists():
            print(
                f"[ingest] Download failed ({exc}); "
                f"falling back to local file at {data_path}"
            )
            return data_path
        raise RuntimeError(
            "Dataset download failed and no local copy is available."
        ) from exc


def main() -> None:
    """CLI entrypoint."""
    config = load_config()
    data_path = ensure_data_available(config)
    print(f"[ingest] Data ready at {data_path}")


if __name__ == "__main__":
    main()
