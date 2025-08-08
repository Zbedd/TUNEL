"""
TUNEL Quantification Package

This package provides tools for analyzing TUNEL-stained tissue sections,
including nuclei segmentation, classification, and statistical analysis.
"""
from __future__ import annotations
from pathlib import Path
import yaml

def load_defaults(path: Path | None = None) -> dict:
    """
    Load default configuration from YAML file.
    
    Args:
        path: Optional path to config file. If None, uses config/default.yaml
        
    Returns:
        Dictionary containing configuration settings
    """
    if path is None:
        # Look for config file relative to package location
        path = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    
    with open(path, "r") as fh:
        return yaml.safe_load(fh)

# Load configuration once at import time
DEFAULTS = load_defaults()