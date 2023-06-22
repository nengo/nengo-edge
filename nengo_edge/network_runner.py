"""
Interface for running an exported NengoEdge model on network accessible devices.

Nengo-edge supports running on the Coral dev board via this runner.
"""

import subprocess
from pathlib import Path
from typing import List, Union

import numpy as np

from nengo_edge import config
from nengo_edge.device_modules import coral_device, np_mfcc


class NetworkRunner:
    """
    Run an exported model from nengo edge that runs on remote network devices.

    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory containing the nengo edge export artifacts.
    device_runner : Union[str, Path]
        Path to the file that will run on the device.
    username : str
        Username on the remote device
    hostname : str
        Hostname of the remote device
    """

    def __init__(
        self,
        directory: Union[str, Path],
        device_runner: Union[str, Path],
        username: str,
        hostname: str,
    ):
        self.directory = Path(directory)
        self.device_runner = Path(device_runner)
        self.model_params, self.preprocessing = config.load_params(self.directory)
        self.return_sequences = self.model_params["return_sequences"]

        self.username = username
        self.hostname = hostname
        self.address = f"{self.username}@{self.hostname}"
        self.check_connection()

        self.remote_dir = Path("/tmp/nengo-edge-runner")

        self.reset()

    def check_connection(self) -> None:
        """
        Run ssh to confirm connection to remote device.

        Throws runtime error if connection fails.
        """
        try:
            subprocess.run(f"ssh {self.address} echo ok".split(), check=True)
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to address {self.address}: {e!r}"
            ) from e

    def _scp(
        self, files_to_copy: List[Path]
    ) -> None:  # pragma: no cover (needs device)
        """One liner to send specified files to remote device location."""
        cmd = (
            f"scp {' '.join(str(p) for p in files_to_copy)} "
            f"{self.address}:{self.remote_dir}"
        )

        subprocess.run(cmd.split(), check=True)

    def reset(self) -> None:
        """Reset the state of the network runner."""
        self.prepared = False

    def send_inputs(self, inputs: np.ndarray) -> None:
        """Saves inputs to file and sends to device."""

        # save inputs to file
        filepath = self.directory / "inputs.npz"
        np.savez_compressed(filepath, inputs=inputs)

        # copy to device
        self._scp([filepath])

    def prepare_device_runner(self) -> None:  # pragma: no cover (needs device)
        """Send required runtime parameters/modules before any inputs."""

        subprocess.run(
            f"ssh {self.address} mkdir -p {self.remote_dir}".split(), check=True
        )

        # copy files to remote
        self._scp(
            [
                self.device_runner,
                self.directory / "model_edgetpu.tflite",
                self.directory / "parameters.json",
                Path(np_mfcc.__file__),
            ]
        )
        self.prepared = True

    def _run_model(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:  # pragma: no cover (needs device)
        """Run the main model logic on the given inputs."""
        self.send_inputs(inputs)
        subprocess.run(
            f"ssh {self.address} python3 {self.remote_dir / self.device_runner.name} "
            f"--directory {self.remote_dir} "
            f"{'--return-sequences' if self.return_sequences else ''}".split(),
            check=True,
        )

        # copy outputs back to host
        subprocess.run(
            f"scp {self.address}:{self.remote_dir / 'outputs.npy'} "
            f"{self.directory}".split(),
            check=True,
        )

        outputs = np.load(self.directory / "outputs.npy")
        return outputs

    def run(self, inputs: np.ndarray) -> np.ndarray:  # pragma: no cover (needs device)
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
        if not self.prepared:
            self.prepare_device_runner()

        outputs = self._run_model(inputs)
        return outputs


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
    ):
        super().__init__(
            directory=directory,
            username=username,
            hostname=hostname,
            device_runner=coral_device.__file__,
        )
