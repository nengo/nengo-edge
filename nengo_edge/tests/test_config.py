# pylint: disable=missing-docstring

from pathlib import Path

import pytest

from nengo_edge import config


def test_runner_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find parameter file"):
        config.load_params(tmp_path)
