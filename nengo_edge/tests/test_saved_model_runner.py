# pylint: disable=missing-docstring

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_hw import gpu
from nengo_edge_models.kws.models import lmu_small
from nengo_edge_models.models import MFCC

from nengo_edge.saved_model_runner import SavedModelRunner


@pytest.mark.parametrize("mode", ("model-only", "feature-only", "full"))
def test_runner(
    mode: Literal["model-only", "feature-only", "full"],
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = lmu_small()
    if mode == "feature-only":
        pipeline.model = []
    elif mode == "model-only":
        pipeline.pre = []

    interface = gpu.host.Interface(pipeline, build_dir=tmp_path)

    inputs = rng.uniform(
        -1, 1, size=(32,) + ((49, 20) if mode == "model-only" else (16000,))
    ).astype("float32")

    output0 = interface.run(inputs)

    interface.export_model(tmp_path)

    assert interface.binary_path is not None
    runner = SavedModelRunner(tmp_path)

    output1 = runner.run(inputs)

    assert np.allclose(output0, output1), np.max(np.abs(output0 - output1))


def test_runner_streaming(
    rng: np.random.RandomState, seed: int, tmp_path: Path
) -> None:
    tf.keras.utils.set_random_seed(seed)

    interface = gpu.host.Interface(lmu_small(), build_dir=tmp_path)
    assert isinstance(interface.pipeline.pre[0], MFCC)

    # 3200 timesteps is equivalent to 200ms at 16 kHz
    inputs = rng.uniform(-1, 1, size=(32, 3200)).astype("float32")
    output0 = interface.run(inputs)

    interface.export_model(tmp_path / "streaming", streaming=True)
    runner = SavedModelRunner(tmp_path / "streaming")

    # check that running in parts produces the same output
    stream_chunk_size = inputs.shape[1] // 4
    for i in range(4):
        output1 = runner.run(
            inputs[:, i * stream_chunk_size : (i + 1) * stream_chunk_size]
        )

    assert np.allclose(output0, output1, atol=1e-4), np.max(np.abs(output0 - output1))

    # check that resetting state works
    runner.reset_state()
    for i in range(4):
        output2 = runner.run(
            inputs[:, i * stream_chunk_size : (i + 1) * stream_chunk_size]
        )

    assert np.allclose(output0, output2, atol=1e-4), np.max(np.abs(output0 - output2))

    # test zero padding
    runner.reset_state()
    pad_output = runner.run(inputs[:, :10])
    pad_output_gt = interface.run(
        np.concatenate(
            [
                inputs[:, :10],
                np.zeros((32, interface.pipeline.pre[0].window_size_samples - 10)),
            ],
            axis=1,
        )
    )
    assert np.allclose(pad_output, pad_output_gt, atol=2e-6), np.max(
        np.abs(pad_output - pad_output_gt)
    )
    pad_output_step = runner.run(inputs[:, -10:])
    pad_output_step_gt = interface.run(
        np.concatenate(
            [
                inputs[:, :10],
                np.zeros((32, interface.pipeline.pre[0].window_size_samples - 10)),
                inputs[:, -10:],
                np.zeros((32, interface.pipeline.pre[0].window_stride_samples - 10)),
            ],
            axis=1,
        )
    )
    assert np.allclose(pad_output_step, pad_output_step_gt, atol=2e-6), np.max(
        np.abs(pad_output_step - pad_output_step_gt)
    )


def test_runner_state(tmp_path: Path) -> None:
    pipeline = lmu_small()
    interface = gpu.host.Interface(pipeline, build_dir=tmp_path)
    interface.export_model(tmp_path, streaming=True)

    runner = SavedModelRunner(tmp_path)

    assert runner.state is None
    runner.run(np.ones((3, 100)))
    assert runner.state is not None
    assert len(runner.state) == 9

    with pytest.raises(ValueError, match="does not match saved state batch size"):
        runner.run(np.ones((5, 100)))

    runner.reset_state()
    assert runner.state is None
    runner.run(np.ones((5, 100)))
    assert runner.state is not None
    assert len(runner.state) == 9
