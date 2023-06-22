# pylint: disable=missing-docstring

import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pytest

from nengo_edge import CoralRunner
from nengo_edge.device_modules import np_mfcc


class MockRunner(CoralRunner):
    def _scp(self, files_to_copy: List[Path]) -> None:
        subprocess.run(
            f"cp -r {' '.join(str(p) for p in files_to_copy)} "
            f"{self.remote_dir}".split(),
            check=True,
        )

    def check_connection(self) -> None:
        pass

    def prepare_device_runner(self) -> None:
        # this is the same as the super implementation, but with the ssh mkdir replaced
        # with a local mkdir
        # subprocess.run(
        #     f"ssh {self.address} mkdir -p {self.remote_dir}".split(), check=True
        # )
        self.remote_dir.mkdir(exist_ok=True, parents=True)

        self._scp(
            [
                self.device_runner,
                self.directory / "model_edgetpu.tflite",
                self.directory / "parameters.json",
                Path(np_mfcc.__file__),
            ]
        )
        self.prepared = True


def test_coral_runner(
    monkeypatch: pytest.MonkeyPatch,
    param_dir: Path,
    rng: np.random.RandomState,
) -> None:
    binary_path = param_dir / "model_edgetpu.tflite"
    binary_path.touch()

    with pytest.raises(RuntimeError, match="Cannot connect to address"):
        CoralRunner(directory=param_dir, username="user", hostname="host")

    net_runner = MockRunner(directory=param_dir, username="user", hostname="host")

    batch_size = 3
    inputs = rng.uniform(-0.5, 0.5, size=(batch_size, 16000)).astype("float32")

    # get output from cli
    assert isinstance(net_runner.remote_dir, Path)
    net_runner.prepare_device_runner()

    assert (net_runner.remote_dir / "parameters.json").exists()
    assert (net_runner.remote_dir / "np_mfcc.py").exists()

    net_runner.send_inputs(inputs)
    assert (net_runner.remote_dir / "inputs.npz").exists()
