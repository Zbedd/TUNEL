from __future__ import annotations
from pathlib import Path
import yaml

def load_defaults(path: Path | None = None) -> dict:
    """Load default YAML configuration."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

DEFAULTS = load_defaults()