# pylint: disable=missing-docstring

from pathlib import Path

import pytest

from nengo_edge import CoralRunner
from nengo_edge.tests.test_saved_model_runner import new_tokenizer


@pytest.fixture(scope="module", name="model_path")
def fixture_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("sentencepiece")
    _, tokenizer_path = new_tokenizer(tmp_path)
    return tokenizer_path


def test_coral_runner(param_dir: Path) -> None:
    binary_path = param_dir / "model_edgetpu.tflite"
    binary_path.touch()

    with pytest.raises(RuntimeError, match="Cannot connect to address"):
        CoralRunner(directory=param_dir, username="user", hostname="host")

    net_runner = CoralRunner(
        directory=param_dir, username="user", hostname="host", local=True
    )

    assert isinstance(net_runner.remote_dir, Path)
    assert (net_runner.remote_dir / "parameters.json").exists()
    assert (net_runner.remote_dir / "np_mfcc.py").exists()
