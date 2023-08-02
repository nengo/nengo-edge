"""Interface for loading parameter file associated with a compiled model and runner."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import packaging

from nengo_edge.version import version


def load_params(directory: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load parameters from file."""

    param_path = directory / "parameters.json"
    if not param_path.exists():
        raise FileNotFoundError(f"Could not find parameter file ({param_path})")
    with open(param_path, "r", encoding="utf-8") as f:
        params = json.load(f)
        model_params = params["model"]
        preprocessing = params["preprocessing"]

    check_params(params)
    return model_params, preprocessing


def check_params(params: Dict[str, Any]) -> None:
    """Perform validation checks on parameters."""

    # Warn user if the nengo-edge version that was used when hardware artifacts were
    # compiled does not match the user's local nengo-edge version
    expected_version = params["version"]["nengo-edge"]
    if packaging.version.parse(expected_version) != packaging.version.parse(version):
        local_version = packaging.version.parse(version)
        warnings.warn(
            f"Downloaded model uses nengo-edge "
            f"{expected_version}, but you're using nengo-edge {local_version}. "
            f"Mismatch may cause errors."
        )
