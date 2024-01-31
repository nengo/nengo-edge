# pylint: disable=missing-docstring

import warnings
from pathlib import Path

import pytest

from nengo_edge import config, version


def test_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find parameter file"):
        config.load_params(tmp_path)


def test_hw_artifact_compatibility() -> None:
    # assert warning case
    expected_hw_version = "0.0.0"
    with pytest.warns(UserWarning, match="Mismatch may cause errors."):
        config.check_params({"version": {"nengo-edge": expected_hw_version}})

    # assert no warning, passing case
    expected_hw_version = version.version
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config.check_params({"version": {"nengo-edge": expected_hw_version}})
