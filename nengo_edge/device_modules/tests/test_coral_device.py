# pylint: disable=missing-docstring

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import numpy as np
import pytest

from nengo_edge import network_runner
from nengo_edge.device_modules import coral_device, np_mfcc
from nengo_edge.tests.test_network_runner import MockRunner


class MockDeviceRunner(coral_device.CoralDeviceRunner):
    def __init__(  # pylint: disable=super-init-not-called
        self, directory: Union[str, Path], return_sequences: bool
    ) -> None:
        self.directory = Path(directory)
        self.return_sequences = return_sequences

    def run(self, inputs: np.ndarray) -> np.ndarray:
        return inputs


def mock_feature_extractor(
    args: argparse.Namespace,
) -> np_mfcc.LogMelFeatureExtractor:
    # note: this is the same as coral_device._build_feature_extractor,
    # with this import removed
    # from np_mfcc import LogMelFeatureExtractor

    with open(Path(args.directory) / "parameters.json", "r", encoding="utf-8") as f:
        parameters = json.load(f)

    options = parameters["preprocessing"]
    feature_extractor = np_mfcc.LogMelFeatureExtractor(**options)
    return feature_extractor


def test_cli(
    monkeypatch: pytest.MonkeyPatch,
    param_dir: Path,
    rng: np.random.RandomState,
) -> None:
    batch_size = 3
    inputs = rng.uniform(-0.5, 0.5, size=(batch_size, 16000)).astype("float32")

    monkeypatch.setattr(network_runner, "CoralRunner", MockRunner)
    monkeypatch.setattr(coral_device, "CoralDeviceRunner", MockDeviceRunner)
    monkeypatch.setattr(
        coral_device, "_build_feature_extractor", mock_feature_extractor
    )

    binary_path = param_dir / "model_edgetpu.tflite"
    binary_path.touch()

    runner = MockRunner(directory=param_dir, username="user", hostname="hostname")

    # get output from cli
    assert isinstance(runner.remote_dir, Path)
    runner.prepare_device_runner()
    runner.send_inputs(inputs)

    monkeypatch.setattr(
        sys,
        "argv",
        f"coral_runner.py --directory {runner.remote_dir}".split(),
    )
    coral_device.cli()

    cli_output = np.load(runner.remote_dir / "outputs.npy")

    options = runner.preprocessing
    feature_extractor = np_mfcc.LogMelFeatureExtractor(**options)
    features = feature_extractor(inputs)

    assert np.all(cli_output == features)
