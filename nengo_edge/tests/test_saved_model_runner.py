# pylint: disable=missing-docstring

import json
from pathlib import Path
from string import ascii_lowercase
from typing import Any, Dict, Literal, Tuple

import numpy as np
import pytest
import tensorflow as tf
from nengo_edge_hw import gpu
from nengo_edge_models.asr.metrics import decode_predictions
from nengo_edge_models.asr.models import asr_tiny
from nengo_edge_models.kws.models import kws_small
from nengo_edge_models.layers import Tokenizer
from nengo_edge_models.models import Tokenizer as TokenizerDesc
from nengo_edge_models.nlp.models import LMUformerEncoderNLP, nlp_sim_small

from nengo_edge import ragged, version
from nengo_edge.saved_model_runner import SavedModelRunner


def new_tokenizer(tmp_path: Path) -> Tuple[Tokenizer, Path]:
    tokenizer = Tokenizer(
        vocab_size=256,
        corpus=Path(__file__).read_text(encoding="utf-8").splitlines(),
    )
    (tmp_path / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer_path = tokenizer.save(tmp_path / "tokenizer")
    return tokenizer, tokenizer_path


@pytest.mark.parametrize("mode", ("model-only", "feature-only", "full"))
def test_runner(
    mode: Literal["model-only", "feature-only", "full"],
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = kws_small()
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

    pipeline = asr_tiny() if model_type == "asr" else nlp_sim_small()
    _, tokenizer_path = new_tokenizer(tmp_path)

    if model_type == "asr":
        assert isinstance(pipeline.post[-1], TokenizerDesc)
        pipeline.post[-1].tokenizer_file = tokenizer_path
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

        assert np.allclose(ragged_out[0][None, ...], ragged_out0, atol=4e-6), np.max(
            abs(ragged_out[0] - ragged_out0)
        )
        assert np.allclose(ragged_out[1][None, ...], ragged_out1, atol=4e-6), np.max(
            abs(ragged_out[1] - ragged_out1)
        )


def test_asr_detokenization(
    rng: np.random.RandomState,
    seed: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tf.keras.utils.set_random_seed(seed)

    pipeline = asr_tiny()
    tokenizer, tokenizer_path = new_tokenizer(tmp_path)
    assert isinstance(pipeline.post[-1], TokenizerDesc)
    pipeline.post[-1].tokenizer_file = tokenizer_path

    interface = gpu.host.Interface(pipeline, build_dir=tmp_path, return_sequences=True)
    interface.export_model(tmp_path)

    monkeypatch.setattr(SavedModelRunner, "_run_model", lambda s, x: x)
    runner = SavedModelRunner(tmp_path)

    inputs = rng.uniform(0, 1, (32, 100, 257))
    # ensure some blank tokens are in the sequences
    # as well as repeats
    for i in range(0, 32):
        # repeats
        inputs[i, i : i + 2] = i
        # blank at the end of sequence
        inputs[i, -1] = 256

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

    pipeline = nlp_sim_small()
    tokenizer, tokenizer_path = new_tokenizer(tmp_path)

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


def test_runner_error_warnings(tmp_path: Path) -> None:
    # test warning for asr models when no tokenizer file is present
    params: Dict[str, Dict[str, Any]] = {
        "preprocessing": {"tokenizer_file": "does not exist"},
        "postprocessing": {"tokenizer_file": "does not exist"},
        "version": {"nengo-edge": version.version},
        "model": {"type": "asr", "return_sequences": True},
    }

    interface = gpu.host.Interface(
        kws_small(), build_dir=tmp_path, return_sequences=True
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
