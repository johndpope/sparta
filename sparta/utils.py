"""Shared utilities for the SPARTA package."""

import json
import gc
import logging
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def cleanup_resources(*model_refs):
    """Release GPU memory for given model references."""
    for ref in model_refs:
        if ref is not None:
            del ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(data: Any, path: str | Path) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved JSON to {path}")


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
