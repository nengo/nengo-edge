# pylint: disable=missing-docstring

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_hw import coral
from nengo_edge_models.kws.tests.test_models import _test_lmu

from nengo_edge.tflite_runner import TFLiteRunner


@pytest.mark.parametrize("return_sequences", (True, False))
def test_runner_streaming(
    return_sequences: bool,
    tmp_path: Path,
    rng: np.random.RandomState,
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
) -> None:
    batch_size = 3
    tf.keras.utils.set_random_seed(seed)

    # export a test model
    monkeypatch.setattr(coral.host.Interface, "io_dtype", None)
    host = coral.host.Interface(
        _test_lmu(),
        build_dir=tmp_path,
        use_device=False,
        return_sequences=return_sequences,
    )

    host.export_model(tmp_path / "exported")

    inputs = rng.uniform(-0.5, 0.5, size=(batch_size, 16000))

    runner = TFLiteRunner(tmp_path / "exported")

    # check that batched and non-batched models produce the same output (state is being
    # properly reset between batch items)
    out = []
    for i in range(batch_size):
        out.append(runner.run(inputs[i : i + 1]))
        runner.reset_state(batch_size=1)
    out0 = np.concatenate(out, axis=0)

    runner.reset_state(batch_size=batch_size)
    out1 = runner.run(inputs)

    assert (
        out0.shape
        == out1.shape
        == ((batch_size, 49, 10) if return_sequences else (batch_size, 10))
    )

    # increased tolerance due to float32 variance
    assert np.allclose(out0, out1, atol=1e-5), np.max(abs(out0 - out1))

    # check that running sequentially produces the same output as running all at once
    runner.reset_state(batch_size=batch_size)
    if return_sequences:
        out0 = np.concatenate(
            [runner.run(inputs[:, :4000]), runner.run(inputs[:, 4000:])],
            axis=1,
        )
    else:
        runner.run(inputs[:, :4000])
        out0 = runner.run(inputs[:, 4000:])
    runner.reset_state(batch_size=batch_size)
    out1 = runner.run(inputs)
    assert (
        out0.shape
        == out1.shape
        == ((batch_size, 49, 10) if return_sequences else (batch_size, 10))
    )
    assert np.allclose(out0, out1, atol=5e-5), np.max(abs(out0 - out1))


def test_runner_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_desc = _test_lmu()
    model_desc.n_unroll = 2
    monkeypatch.setattr(coral.host.Interface, "io_dtype", None)
    host = coral.host.Interface(
        model_desc, build_dir=tmp_path, use_device=False, mode="model-only"
    )
    host.export_model(tmp_path)

    runner = TFLiteRunner(tmp_path)

    with pytest.raises(ValueError, match="evenly divided by unroll"):
        runner.run(np.zeros((1, 3, 10)))

    Path(tmp_path / "model.tflite").unlink()

    with pytest.raises(FileNotFoundError, match="Could not find model file"):
        TFLiteRunner(tmp_path)


def test_runner_quantized(tmp_path: Path, rng: np.random.RandomState) -> None:
    model_desc = _test_lmu()
    host = coral.host.Interface(
        model_desc,
        build_dir=tmp_path,
        use_device=False,
        representative_data=rng.uniform(-1, 1, size=(32, 16000)),
    )
    host.export_model(tmp_path)

    runner = TFLiteRunner(tmp_path)

    x = rng.uniform(-1, 1, size=(1, 16000))
    y0 = runner.run(x)
    y1 = host.run(x)

    assert np.allclose(y0, y1, atol=5e-5), np.max(abs(y0 - y1))
