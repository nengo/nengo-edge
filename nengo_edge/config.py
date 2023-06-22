"""Interface for loading parameter file associated with a compiled model and runner."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_params(directory: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load parameters from file."""

    param_path = directory / "parameters.json"
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find parameter file ({param_path})")
    with open(param_path, "r", encoding="utf-8") as f:
        params = json.load(f)
        model_params = params["model"]
        preprocessing = params["preprocessing"]

    return model_params, preprocessing
