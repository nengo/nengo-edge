import json
from typing import Any

import pytest
from click.testing import CliRunner

from nengo_edge.version import version


@pytest.fixture
def param_dir(tmp_path):
    params = {
        "preprocessing": {
            "window_size_ms": 40,
            "window_stride_ms": 20,
            "mel_num_bins": 40,
            "dct_num_features": 20,
            "sample_rate": 16000,
            "mel_lower_edge_hertz": 20,
            "mel_upper_edge_hertz": 7000,
            "log_epsilon": 1e-12,
        },
        "model": {
            "input_shape": [None, 20],
            "output_shape": [None, 10],
            "state_shapes": [10, 10],
            "return_sequences": True,
            "n_unroll": 1,
        },
        "version": {"nengo-edge": version},
    }

    with open(tmp_path / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f)

    return tmp_path


@pytest.fixture
def cli_runner() -> CliRunner:
    """
    A version of ``click.testing.CliRunner`` that echoes output to stdout.

    This is useful as it allows pytest to capture the output like normal (rather
    than the CliRunner eating it all).
    """

    class EchoCliRunner(CliRunner):
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            result = super().invoke(*args, **kwargs)
            print(result.stdout)
            return result

    return EchoCliRunner()


def pytest_sessionstart(session: pytest.Session) -> None:
    """Hook that runs on session start."""

    try:
        import tensorflow as tf  # pylint: disable=unused-import
    except ImportError:  # pragma: no cover
        pytest.exit("Tensorflow is not installed; cannot run tests.")
