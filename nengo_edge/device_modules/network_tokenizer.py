"""Interface for running an exported SentencePiece tokenizer on network accessible
devices."""

# TODO: move this to network_runner.py

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

    def __init__(
        self,
        directory: Union[str, Path],
        username: str,
        hostname: str,
        local: bool = False,
    ):
        super().__init__(
            directory=directory, username=username, hostname=hostname, local=local
        )

        self.tokenizer_file = None
        for params in [self.preprocessing, self.postprocessing]:
            if "tokenizer_file" in params:
                self.tokenizer_file = params["tokenizer_file"]

        if self.tokenizer_file is None:
            raise TypeError("Exported config does not contain any tokenizers")

        # Copy files to remote
        self._scp([self.directory / self.tokenizer_file])

    def tokenize(self, input_text: str) -> List[int]:
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
        output = self._ssh(
            f"spm_encode"
            f" --model={self.remote_dir / self.tokenizer_file}"
            f" --output_format=id",
            std_in=input_text.encode("utf-8"),
        ).decode("utf-8")
        token_string = output.rstrip()
        token_ids = [int(token) for token in token_string.split()]
        return token_ids

    def detokenize(self, inputs: np.ndarray) -> str:
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
        output = self._ssh(
            f"spm_decode"
            f" --model={self.remote_dir / self.tokenizer_file}"
            f" --input_format=id",
            std_in=token_string.encode("utf-8"),
        ).decode("utf-8")
        decoded_text = output.rstrip()
        return decoded_text

    def run(self, inputs: Union[np.ndarray, List[str]]) -> np.ndarray:
        """Run the main tokenizer logic on the given inputs."""

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
