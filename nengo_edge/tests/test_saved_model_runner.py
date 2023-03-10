# pylint: disable=missing-docstring

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_hw import gpu
from nengo_edge_models.tests.test_models import _test_lmu

from nengo_edge.saved_model_runner import Runner


@pytest.mark.parametrize("mode", ("model-only", "feature-only", "full"))
def test_runner(
    mode: Literal["model-only", "feature-only", "full"],
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    interface = gpu.host.Interface(_test_lmu(), mode=mode, build_dir=tmp_path)

    inputs = rng.uniform(
        -1, 1, size=(32,) + ((49, 20) if mode == "model-only" else (16000,))
    ).astype("float32")

    output0 = interface.run(inputs)

    interface.export_model(tmp_path)

    assert interface.binary_path is not None
    runner = Runner(tmp_path)

    output1 = runner.run(inputs)

    assert np.allclose(output0, output1), np.max(np.abs(output0 - output1))


def test_runner_streaming(
    rng: np.random.RandomState, seed: int, tmp_path: Path
) -> None:
    tf.keras.utils.set_random_seed(seed)

    interface = gpu.host.Interface(_test_lmu(), mode="full", build_dir=tmp_path)

    inputs = rng.uniform(-1, 1, size=(32, 16000)).astype("float32")

    output0 = interface.run(inputs)

    interface.export_model(tmp_path / "streaming")

    runner = Runner(tmp_path / "streaming")

    # check that running in parts produces the same output
    for i in range(4):
        output1 = runner.run(inputs[:, i * 4000 : (i + 1) * 4000])

    assert np.allclose(output0, output1, atol=1e-5), np.max(np.abs(output0 - output1))

    # check that resetting state works
    runner.reset_state()
    for i in range(4):
        output2 = runner.run(inputs[:, i * 4000 : (i + 1) * 4000])

    assert np.allclose(output0, output2, atol=1e-5), np.max(np.abs(output0 - output2))

    # test zero padding
    runner.reset_state()
    pad_output = runner.run(inputs[:, :10])
    pad_output_gt = (
        interface.run(
            np.concatenate(
                [
                    inputs[:, :10],
                    np.zeros((32, interface.audio_options.window_size_samples - 10)),
                ],
                axis=1,
            )
        ),
    )
    assert np.allclose(pad_output, pad_output_gt), np.max(
        np.abs(pad_output - pad_output_gt)
    )
    pad_output_step = runner.run(inputs[:, -10:])
    pad_output_step_gt = interface.run(
        np.concatenate(
            [
                inputs[:, :10],
                np.zeros((32, interface.audio_options.window_size_samples - 10)),
                inputs[:, -10:],
                np.zeros((32, interface.audio_options.window_stride_samples - 10)),
            ],
            axis=1,
        )
    )
    assert np.allclose(pad_output_step, pad_output_step_gt, atol=2e-6), np.max(
        np.abs(pad_output_step - pad_output_step_gt)
    )
