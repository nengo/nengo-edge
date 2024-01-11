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
