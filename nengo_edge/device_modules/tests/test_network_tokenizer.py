# pylint: disable=missing-docstring

import json
import shutil
import subprocess
from pathlib import Path
from string import ascii_lowercase
from typing import Any, Dict, List

import numpy as np
import pytest
from tensorflow_text import SentencepieceTokenizer

from nengo_edge.device_modules.network_tokenizer import NetworkTokenizer
from nengo_edge.tests.test_saved_model_runner import new_tokenizer
from nengo_edge.version import version


@pytest.fixture(scope="module", name="model_path")
def fixture_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("sentencepiece")
    _, tokenizer_path = new_tokenizer(tmp_path)
    return tokenizer_path


def prepare_tokenizer(
    temp_dir: Path,
    sp_model_path: Path,
    layer: str = "preprocessing",
    model_type: str = "asr",
) -> None:
    params: Dict[str, Dict[str, Any]] = {
        "preprocessing": {},
        "model": {
            "return_sequences": True,
            "type": model_type,
        },
        "postprocessing": {},
        "version": {"nengo-edge": version},
    }
    if layer in params:
        params[layer]["tokenizer_file"] = sp_model_path.name

    with open(temp_dir / "parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f)

    shutil.copy(sp_model_path, temp_dir / sp_model_path.name)


class MockTokenizer(NetworkTokenizer):
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
        assert self.tokenizer_file is not None
        self.remote_dir.mkdir(exist_ok=True, parents=True)
        self._scp([self.directory / self.tokenizer_file])
        self.prepared = True

        self.sp = SentencepieceTokenizer(
            (self.remote_dir / self.tokenizer_file).read_bytes()
        )

    def tokenize(self, input_text: str) -> List[int]:
        tokens = self.sp.tokenize(input_text).numpy()
        return tokens

    def detokenize(self, inputs: np.ndarray) -> str:
        decoded_text = self.sp.detokenize(inputs[inputs != -1]).numpy().decode()
        return decoded_text


@pytest.mark.parametrize("layer", ("preprocessing", "postprocessing"))
def test_paths(layer: str, model_path: Path, tmp_path: Path) -> None:
    prepare_tokenizer(tmp_path, sp_model_path=model_path, model_type="asr", layer=layer)
    network_tokenizer = MockTokenizer(tmp_path, "name", "host")
    network_tokenizer.prepare_device_runner()
    assert network_tokenizer.tokenizer_file is not None
    assert network_tokenizer.tokenizer_file == model_path.name
    assert (network_tokenizer.remote_dir / model_path.name).exists()


def test_errors(rng: np.random.RandomState, model_path: Path, tmp_path: Path) -> None:
    # test wrong model type error
    prepare_tokenizer(tmp_path, sp_model_path=model_path, model_type="kws")
    with pytest.raises(ValueError, match="only supported for ASR and NLP models"):
        MockTokenizer(tmp_path, "name", "host")

    # test error with missing tokenizer_file entry in the parameter file
    prepare_tokenizer(tmp_path, sp_model_path=model_path, layer="no layer")
    with pytest.raises(ValueError, match="Cannot find entry for tokenizer_file"):
        MockTokenizer(tmp_path, "name", "host")

    # test wrong input type on detokenization
    prepare_tokenizer(tmp_path, sp_model_path=model_path)
    network_tokenizer = MockTokenizer(tmp_path, "name", "host")
    with pytest.raises(ValueError, match="must be one of int32/int64"):
        network_tokenizer.run(rng.randint(0, 255, (32, 50)).astype("float32"))

    # test wrong number of dimension on input
    with pytest.raises(ValueError, match="inputs must have exactly 2 dimensions"):
        network_tokenizer.run(rng.randint(0, 255, (32,)).astype("int32"))


def test_tokenize_detokenize(
    rng: np.random.RandomState, param_dir: Path, model_path: Path, tmp_path: Path
) -> None:
    prepare_tokenizer(tmp_path, sp_model_path=model_path)
    network_tokenizer = MockTokenizer(tmp_path, "name", "host")

    input_text = [
        " ".join(
            "".join(rng.choice([*ascii_lowercase], size=3))  # type: ignore
            for _ in range(10)
        )
        for i in range(32)
    ]
    token_ids = network_tokenizer.run(input_text)
    max_len = np.max([t.shape[0] for t in token_ids])
    token_ids = np.array(
        [
            np.concatenate([t, np.full(max_len - t.shape[0], -1, dtype="int32")])
            for t in token_ids
        ]
    )
    decoded_text = network_tokenizer.run(token_ids)

    assert all(t == t_decoded for t, t_decoded in zip(input_text, decoded_text))
