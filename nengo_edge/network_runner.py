"""
Interface for running an exported NengoEdge model on network accessible devices.

Nengo-edge supports running on the Coral dev board via this runner.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from nengo_edge import config
from nengo_edge.device_modules import coral_device, np_mfcc


class NetworkRunner:
    """
    Run an exported component on a remote network device.

    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory containing the NengoEdge export artifacts.
    username : str
        Username on the remote device
    hostname : str
        Hostname of the remote device
    local : bool
        If True, run locally rather than over the network (for debugging purposes).
    """

    def __init__(
        self,
        directory: Union[str, Path],
        username: str,
        hostname: str,
        local: bool = False,
    ):
        self.directory = Path(directory)
        self.model_params, self.preprocessing, self.postprocessing = config.load_params(
            self.directory
        )
        self.username = username
        self.hostname = hostname
        self.address = f"{self.username}@{self.hostname}"
        self.local = local

        try:
            self._ssh("echo ok")
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to address {self.address}: {e!r}"
            ) from e

        if local:
            self._tmp_dir = (
                tempfile.TemporaryDirectory(  # pylint: disable=consider-using-with
                    dir=directory
                )
            )
            self.remote_dir = Path(self._tmp_dir.name)
        else:  # pragma: no cover (needs device)
            self.remote_dir = Path(f"/tmp/nengo-edge-{type(self).__name__.lower()}")
            self._ssh(f"rm -rf {self.remote_dir}")
            self._ssh(f"mkdir -p {self.remote_dir}")

    def _scp(self, files_to_copy: List[Path]) -> None:
        """One liner to send specified files to remote device location."""
        cmd = (
            (["cp", "-r"] if self.local else ["scp"])
            + [str(p) for p in files_to_copy]
            + [
                (
                    str(self.remote_dir)
                    if self.local
                    else f"{self.address}:{self.remote_dir}"
                )
            ]
        )

        subprocess.run(cmd, check=True)

    def _ssh(self, cmd: str, std_in: Optional[bytes] = None) -> bytes:
        """Run a command over ssh."""
        return subprocess.run(
            cmd if self.local else ["ssh", self.address, cmd],
            input=std_in,
            check=True,
            stdout=subprocess.PIPE,
            shell=self.local,
        ).stdout


class CoralRunner(NetworkRunner):
    """
    Run a model exported from NengoEdge on the Coral board.

    See `NetworkRunner` for parameter descriptions.
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

        self.device_runner = Path(coral_device.__file__)
        self.return_sequences = self.model_params["return_sequences"]

        # copy files to remote
        self._scp(
            [
                self.device_runner,
                self.directory / "model_edgetpu.tflite",
                self.directory / "parameters.json",
                Path(np_mfcc.__file__),
            ]
        )

    def run(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:  # pragma: no cover (needs device)
        """
        Run model inference on a batch of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values (must have shape ``(batch_size, input_steps)``)

        Returns
        -------
        outputs : ``np.ndarray``
            Model output values (with shape ``(batch_size, output_d)`` if
            ``self.model_params['return_sequences']=False``
            else ``(batch_size, output_steps, output_d)``).
        """

        # Save inputs to file
        filepath = self.directory / "inputs.npz"
        np.savez_compressed(filepath, inputs=inputs)

        # Copy to device
        self._scp([filepath])

        # Run model on device
        self._ssh(
            f"python3 {self.remote_dir / self.device_runner.name} "
            f"--directory {self.remote_dir} "
            f"{'--return-sequences' if self.return_sequences else ''}"
        )

        # Copy outputs back to host
        subprocess.run(
            f"scp {self.address}:{self.remote_dir / 'outputs.npy'} "
            f"{self.directory}".split(),
            check=True,
        )

        outputs = np.load(self.directory / "outputs.npy")
        return outputs


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
            outputs: np.ndarray = np.asarray(
                [self.tokenize(text) for text in inputs], dtype=object
            )
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
