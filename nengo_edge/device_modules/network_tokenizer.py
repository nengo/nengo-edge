"""Interface for running an exported SentencePiece tokenizer on network accessible
devices."""

import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np

from nengo_edge.network_runner import NetworkRunner


class NetworkTokenizer(NetworkRunner):
    """
    Class to access a SentencePiece command line interface installed on a network
    accessible device.

    Can be used to tokenize and detokenize values for asr and nlp inference.
    """

    def __init__(self, directory: Union[str, Path], username: str, hostname: str):
        super().__init__(
            directory=directory, device_runner="", username=username, hostname=hostname
        )

        if self.model_params["type"] not in ["asr", "nlp"]:
            raise ValueError("Tokenizers are only supported for ASR and NLP models.")

        self.tokenizer_file = None
        for params in [self.preprocessing, self.postprocessing]:
            if "tokenizer_file" in params:
                self.tokenizer_file = params["tokenizer_file"]

        if self.tokenizer_file is None:
            raise ValueError(
                f"Cannot find entry for tokenizer_file in "
                f"{self.directory}/parameters.json."
            )

        self.remote_dir = Path("/tmp/nengo-edge-tokenizer")

    def prepare_device_runner(self) -> None:  # pragma: no cover (needs device)
        """Send required runtime parameters/modules before any inputs."""
        assert (
            len(self.remote_dir.parts) > 2 and self.remote_dir.parts[1] == "tmp"
        ), f"remote_dir ({self.remote_dir}) not in /tmp"

        subprocess.run(
            f"ssh {self.address} rm -rf {self.remote_dir}".split(), check=True
        )
        subprocess.run(
            f"ssh {self.address} mkdir -p {self.remote_dir}".split(), check=True
        )

        # copy files to remote
        assert self.tokenizer_file is not None
        self._scp([self.directory / self.tokenizer_file])
        self.prepared = True

    def tokenize(self, input_text: str) -> List[int]:  # pragma: no cover (needs device)
        """
        Map strings to their corresponding integer tokens.

        This function utilizes an ssh command to access the SentencePiece command
        line interface on the configured network device.

        Parameters
        ----------
        input_text: str
            Input string to be tokenized.

        Returns
        -------
        token_ids: List[int]
            A list of integers of length ``(n_tokens)``.
        """
        assert self.tokenizer_file is not None
        cmd = (
            f"ssh {self.address} spm_encode"
            f" --model={self.remote_dir / self.tokenizer_file}"
            f" --output_format=id"
        )
        output = subprocess.run(
            cmd.split(),
            input=input_text,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        token_string = output.stdout.rstrip()
        token_ids = [int(token) for token in token_string.split()]
        return token_ids

    def detokenize(self, inputs: np.ndarray) -> str:  # pragma: no cover (needs device)
        """
        Map integer tokens to their corresponding string token.

        This function utilizes an ssh command to access the SentencePiece command
        line interface on the configured network device.

        Parameters
        ----------
        inputs: np.ndarray
            Input array containing integer values. Array should be generated
            from a top-1 decoding strategy (e.g. greedy decoding)
            on asr model outputs and have a size of ``(batch_size, output_steps)``.

        Returns
        -------
        decoded_text: str
            A string generated from the decoded input integers.
        """

        token_string = " ".join([str(token) for token in inputs[inputs != -1]])

        assert self.tokenizer_file is not None
        cmd = (
            f"ssh {self.address} spm_decode"
            f" --model={self.remote_dir / self.tokenizer_file}"
            f" --input_format=id"
        )
        output = subprocess.run(
            cmd.split(),
            input=token_string,
            encoding="utf-8",
            capture_output=True,
            check=True,
        )
        decoded_text = output.stdout.rstrip()
        return decoded_text

    def run(self, inputs: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Run the main tokenizer logic on the given inputs."""
        if not self.prepared:
            self.prepare_device_runner()

        if isinstance(inputs[0], str):
            outputs = np.asarray([self.tokenize(text) for text in inputs], dtype=object)
        else:
            assert isinstance(inputs, np.ndarray)

            if inputs.dtype not in ["int32", "int64"]:
                raise ValueError(f"{inputs.dtype=} must be one of int32/int64.")

            if inputs.ndim != 2:
                raise ValueError(
                    f"inputs must have exactly 2 dimensions, found {inputs.ndim}."
                )

            outputs = np.asarray(
                [self.detokenize(tokens) for tokens in inputs], dtype=np.str_
            )
        return outputs
