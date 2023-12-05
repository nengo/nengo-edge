# pylint: disable=missing-docstring

import json
from pathlib import Path
from string import ascii_lowercase
from typing import Any, Dict, Literal

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_hw import gpu
from nengo_edge_models.asr.metrics import decode_predictions
from nengo_edge_models.asr.models import lmuformer_tiny
from nengo_edge_models.kws.models import lmu_small
from nengo_edge_models.layers import Tokenizer
from nengo_edge_models.models import MFCC
from nengo_edge_models.models import Tokenizer as TokenizerDesc
from nengo_edge_models.nlp.models import LMUformerEncoderNLP, lmuformer_small_sim

from nengo_edge import ragged, version
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
    runner = SavedModelRunner(tmp_path)

    output1 = runner.run(inputs)

    assert np.allclose(output0, output1), np.max(np.abs(output0 - output1))


@pytest.mark.parametrize("mode", ("model-only", "feature-only", "full"))
@pytest.mark.parametrize("model_type", ("asr", "nlp"))
def test_runner_ragged(
    mode: str, model_type: str, rng: np.random.RandomState, seed: int, tmp_path: Path
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = lmuformer_tiny() if model_type == "asr" else lmuformer_small_sim()
    tokenizer = Tokenizer(
        vocab_size=256,
        corpus=Path(__file__).read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")

    if model_type == "asr":
        assert isinstance(pipeline.post[0], TokenizerDesc)
        pipeline.post[0].tokenizer_file = tokenizer_path
    else:
        assert isinstance(pipeline.pre[0], TokenizerDesc)
        assert isinstance(pipeline.model[1], LMUformerEncoderNLP)
        pipeline.pre[0].tokenizer_file = tokenizer_path
        pipeline.pre[0].vocab_size = 256
        pipeline.model[1].vocab_size = 256

    if mode == "feature-only":
        pipeline.model = []
        pipeline.post = []

    elif mode == "model-only":
        if model_type == "asr":
            pipeline.pre = []
        pipeline.post = []

    interface = gpu.host.Interface(pipeline, build_dir=tmp_path, return_sequences=True)
    interface.export_model(tmp_path)
    runner = SavedModelRunner(tmp_path)
    if model_type == "asr":
        inputs = rng.uniform(
            -1, 1, size=(32,) + ((49, 80) if mode == "model-only" else (16000,))
        ).astype("float32")

        inputs = np.array(
            [
                inputs[0, : int(inputs.shape[1] * 0.5)],
                inputs[1, : int(inputs.shape[1] * 0.8)],
            ],
            dtype=object,
        )
        ragged_in0 = inputs[0:1]
        ragged_in1 = inputs[1:2]
    else:
        inputs = np.asarray(
            [
                " ".join(
                    "".join(rng.choice([*ascii_lowercase], size=3))
                    for _ in range(10 + i * 10)
                )
                for i in range(2)
            ]
        )

        ragged_in0 = inputs[[0]]
        ragged_in1 = inputs[[1]]

    ragged_out = runner.run(inputs)
    ragged_out0 = runner.run(ragged_in0)
    ragged_out1 = runner.run(ragged_in1)

    if mode == "full" and model_type == "asr":
        # test string output case
        assert len(ragged_out[0]) == len(ragged_out0[0])
        assert len(ragged_out[1]) == len(ragged_out1[0])
        assert ragged_out[0] == ragged_out0[0]
        assert ragged_out[1] == ragged_out1[0]
    else:
        # test numerical output case
        assert (1,) + ragged_out[0].shape == ragged_out0.shape
        assert (1,) + ragged_out[1].shape == ragged_out1.shape

        assert np.allclose(ragged_out[0][None, ...], ragged_out0, atol=2e-6), np.max(
            abs(ragged_out[0] - ragged_out0)
        )
        assert np.allclose(ragged_out[1][None, ...], ragged_out1, atol=2e-6), np.max(
            abs(ragged_out[1] - ragged_out1)
        )


def test_asr_detokenization(
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = lmuformer_tiny()
    tokenizer = Tokenizer(
        vocab_size=256,
        corpus=Path(__file__).read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")
    pipeline.post[0].tokenizer_file = tokenizer_path

    interface = gpu.host.Interface(pipeline, build_dir=tmp_path, return_sequences=True)
    interface.export_model(tmp_path)

    monkeypatch.setattr(SavedModelRunner, "_run_model", lambda s, x: x)
    runner = SavedModelRunner(tmp_path)

    inputs = rng.uniform(0, 1, (32, 10, 257))
    outputs = runner.run(inputs)
    gt = decode_predictions(ragged.to_masked(inputs), tokenizer)
    assert np.all(outputs == gt)


def test_nlp_tokenization(
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = lmuformer_small_sim()
    tokenizer = Tokenizer(
        vocab_size=256,
        corpus=Path(__file__).read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")

    assert isinstance(pipeline.pre[0], TokenizerDesc)
    assert isinstance(pipeline.model[1], LMUformerEncoderNLP)
    pipeline.pre[0].tokenizer_file = tokenizer_path
    pipeline.pre[0].vocab_size = 256
    pipeline.model[1].vocab_size = 256

    interface = gpu.host.Interface(pipeline, build_dir=tmp_path)
    interface.export_model(tmp_path)

    monkeypatch.setattr(SavedModelRunner, "_run_model", lambda s, x: x)
    runner = SavedModelRunner(tmp_path)

    text_batch = [
        " ".join("".join(rng.choice([*ascii_lowercase], size=3)) for _ in range(10))
        for _ in range(32)
    ]

    outputs = runner.run(text_batch)
    gt = tokenizer.tokenize(text_batch).numpy()  # type: ignore
    assert len(outputs) == len(gt)
    for x, y in zip(outputs, gt):
        np.testing.assert_allclose(x, y)


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


def test_runner_error_warnings(tmp_path: Path) -> None:
    # test warning for asr models when no tokenizer file is present
    params: Dict[str, Dict[str, Any]] = {
        "preprocessing": {"tokenizer_file": "does not exist"},
        "postprocessing": {"tokenizer_file": "does not exist"},
        "version": {"nengo-edge": version.version},
        "model": {"type": "asr", "return_sequences": True},
    }

    interface = gpu.host.Interface(
        lmu_small(), build_dir=tmp_path, return_sequences=True
    )
    interface.export_model(tmp_path)
    # overwrite exported parameters
    with open(tmp_path / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f)

    with pytest.warns(UserWarning, match="cannot decode ASR outputs"):
        SavedModelRunner(tmp_path)

    # test error for nlp models when no tokenizer file is present
    params["model"]["type"] = "nlp"
    with open(tmp_path / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f)

    with pytest.raises(ValueError, match="required to process string inputs"):
        SavedModelRunner(tmp_path)
