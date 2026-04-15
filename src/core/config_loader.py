# core/config_loader.py

from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)