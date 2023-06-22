"""
Runner for running an exported NengoEdge model in TFLite format on microcontroller
devices.

NengoEdge supports interfacing with STM32F746 Discovery and nRF52840 Dev boards
via this runner.
"""
import shutil
import time
import timeit
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import serial
from numpy.typing import NDArray
from rich import progress

from nengo_edge import config


class MicroRunner:
    """
    Run an exported model from nengo edge that runs on micro devices.

    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory containing the nengo edge export artifacts.
    serial_path : Union[str, Path]
        Path to drive where device serial port is mounted
    device_path : Union[str, Path]
        Path to drive where device is mounted
    binary_name : str
        Name for the binary file used by this device.
    """

    # control bytes for serial communication
    SUCCESS = b"S"
    REQ = b"R"
    ACK_NOSTATE = b"N"
    ACK_STATE = b"T"
    FAIL = b"F"

    def __init__(
        self,
        directory: Union[str, Path],
        serial_path: Union[str, Path],
        device_path: Union[str, Path],
        binary_name: str,
    ):
        self.directory = Path(directory)
        self.serial_path = Path(serial_path)
        self.device_path = Path(device_path)
        self.binary_name = binary_name
        self.model_params, self.preprocessing = config.load_params(self.directory)
        self.uses_states = len(self.model_params["state_shapes"]) > 0

        # the amount of data that will be output by the model
        self.recv_size = self.model_params["output_shape"][-1]

        # Flash the device binary
        self.flash_binary()

    def __enter__(self) -> "MicroRunner":
        self.serial_port = serial.Serial(
            str(self.serial_path), baudrate=115200, timeout=30
        )
        time.sleep(0.1)  # Add a slight delay to make sure serial port is ready
        return self

    def __exit__(self, *_: Any) -> None:
        self.serial_port.close()

    def flash_binary(self) -> None:
        """Copy the binary to the device."""

        shutil.copyfile(
            self.directory / self.binary_name, self.device_path / self.binary_name
        )

        # Wait for 10s for device to be programmed with copied binary to avoid
        # race condition.
        for _ in progress.track(range(10), description="Flashing binary to device"):
            time.sleep(1)

    def prime_inputs(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare inputs to match expected audio buffer in the model binary."""

        # We need to prime the audio buffer with window - stride samples so we
        # can properly extract a full window of MFCCs given stride new samples
        n_priming = (
            (
                self.preprocessing["window_size_ms"]
                - self.preprocessing["window_stride_ms"]
            )
            * self.preprocessing["sample_rate"]
            / 1000
        )

        if not np.isclose(n_priming, int(n_priming)):
            raise ValueError(
                f"Sample rate from parameters ({self.preprocessing['sample_rate']})"
                f" is too small. Use sample_rate >> 100."
            )

        n_priming = min(int(n_priming), inputs.shape[1])
        primer_inputs = inputs[:, :n_priming]
        inputs = inputs[:, n_priming:]

        inputs = inputs.reshape(
            (
                inputs.shape[0],
                -1,
                int(
                    self.preprocessing["sample_rate"]
                    * 1e-3
                    * self.preprocessing["window_stride_ms"]
                ),
            )
        )

        return inputs, primer_inputs

    def prepare_serial(self, inputs: np.ndarray) -> None:
        """Send required runtime parameters before any inputs."""

        n_steps = np.shape(inputs)[1]
        n_unroll = self.model_params["input_shape"][0]

        # Make sure data can be pieced into unroll sized chunks
        if n_steps % n_unroll != 0:
            raise ValueError(
                f"Number of timesteps ({n_steps}) cannot be evenly divided by unroll "
                f"size ({n_unroll})"
            )

        if self.model_params["return_sequences"]:
            self.recv_size *= n_unroll

        # Binary expects number of loops it will execute at runtime (sent over UART)
        self.n_hw_laps = len(inputs) * n_steps // n_unroll
        self.send_data_burst(
            serial_port=self.serial_port,
            user_data=np.asarray([float(self.n_hw_laps)]),
            state_data=None,
        )

    def run(self, inputs: np.ndarray) -> np.ndarray:
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
        if len(inputs.shape) != 2:
            raise ValueError("Inputs must have ndim=2 (batch_size, input_steps).")

        inputs, primer_inputs = self.prime_inputs(inputs)
        self.prepare_serial(inputs)

        outputs = self._run_model(inputs, primer_inputs=primer_inputs)

        self.serial_port.reset_input_buffer()

        return outputs

    def _run_model(
        self, inputs: np.ndarray, primer_inputs: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Run the main model logic on the given inputs."""

        n_steps = np.shape(inputs)[1]
        n_unroll = self.model_params["input_shape"][0]

        outputs = []
        with progress.Progress() as bar:
            task = bar.add_task("Running inference on device", total=self.n_hw_laps)
            for index_batch, user_data in enumerate(inputs):
                # Reshape this input to be in unroll sized chunks.
                user_data = np.reshape(user_data, (n_steps // n_unroll, -1))
                # Create list of initial state data for this input (or None if no
                # model)
                state_data = (
                    [np.zeros(shape) for shape in self.model_params["state_shapes"]]
                    if self.uses_states
                    else None
                )

                # Send in audio primer inputs
                if primer_inputs is not None:
                    # We send initial state with the first slice below for
                    # "model-only", so neglect state here and send state below for
                    # consistency.
                    self.send_data_burst(
                        serial_port=self.serial_port,
                        user_data=primer_inputs[index_batch].reshape(-1),
                        state_data=None,
                    )

                # Send each slice over and receive output for that slice
                output_sequences = []
                for index_slice, input_slice in enumerate(user_data):
                    self.send_data_burst(
                        serial_port=self.serial_port,
                        user_data=input_slice,
                        state_data=None
                        if index_slice > 0 or not self.uses_states
                        else state_data,
                    )
                    rxd_data = self.receive_data_burst(
                        serial_port=self.serial_port, size=self.recv_size
                    )

                    if self.model_params["return_sequences"]:
                        # Append all outputs from the past `unroll` timesteps
                        # Need to reshape since they come as block
                        # n_unroll*n_outputs
                        output_sequences.append(
                            rxd_data.reshape(
                                n_unroll,
                                self.recv_size // n_unroll,
                            )
                        )

                    bar.update(task, advance=1)

                # Store final output or all outputs if return_sequences
                outputs.append(
                    # concatenate outputs from each unroll block
                    # note: because the model was built with return_sequences=True,
                    # this represents the output from all the timesteps
                    np.concatenate(output_sequences, axis=0)
                    if self.model_params["return_sequences"]
                    # only use the output from last unroll block
                    # note: because the model was built with return_sequences=False,
                    # the output from the last unroll block is actually the output
                    # from the last timestep
                    else rxd_data
                )

        return np.asarray(outputs)

    @staticmethod
    def send_data_burst(
        serial_port: serial.Serial,
        user_data: np.ndarray,
        state_data: Optional[List[np.ndarray]],
    ) -> None:
        """
        Send the given burst of data over UART to remote device.

        Parameters
        ----------
        serial_port : serial.Serial
            The serial object used to send data
        user_data : np.ndarray
            Model input values (should have shape ``(n_unroll * input_d)``
        state_data : Optional[List[np.ndarray]]
            Initial state values. Each LMU layer has two state values, for the
            hidden (with shape ``(units)``) and memory (with shape
            ``(memory_d * order)``) components. A multi-layer model
            should repeat these in the same order as the layers, i.e.
            ``[memory_state_0, hidden_state_0, memory_state_1, hidden_state_1, ...]``.
        """

        # Only send state if necessary
        flattened_data = user_data
        if state_data is not None:
            flattened_data = np.append(
                flattened_data, np.concatenate(state_data, axis=None)
            )

        timeout = timeit.default_timer() + serial_port.timeout * 5
        reply = b""
        while reply != MicroRunner.SUCCESS and timeit.default_timer() < timeout:
            data = serial_port.read(1)

            if data == MicroRunner.REQ:
                serial_port.write(
                    MicroRunner.ACK_NOSTATE
                    if state_data is None
                    else MicroRunner.ACK_STATE
                )

                serial_port.write(flattened_data.astype(np.float32).tobytes())
                reply = serial_port.read(1)
            else:
                # this will cause the device to re-send the send request
                serial_port.write(MicroRunner.FAIL)

        if reply != MicroRunner.SUCCESS:
            raise TimeoutError("Timed out trying to send data to device")

    @staticmethod
    def receive_data_burst(serial_port: serial.Serial, size: int) -> NDArray:
        """
        Receive burst of data over UART from remote device.

        Parameters
        ----------
        serial_port : serial.Serial
            The serial object used to send data
        size: int
            The count of FP32 numbers to receive from remote device.
        """
        timeout = timeit.default_timer() + serial_port.timeout * 5
        success = False
        while not success and timeit.default_timer() < timeout:
            serial_port.reset_input_buffer()
            serial_port.write(MicroRunner.REQ)

            data = serial_port.read(1)
            if data == MicroRunner.ACK_STATE:
                data = serial_port.read(size * 4)
                if len(data) == size * 4:
                    data = np.frombuffer(data, dtype=np.float32)
                    success = True

            # send response
            serial_port.write(MicroRunner.SUCCESS if success else MicroRunner.FAIL)

        if not success:
            raise TimeoutError("Timed out trying to receive data from device")

        return data


class DiscoRunner(MicroRunner):
    """
    Run a model exported from NengoEdge on the Disco board.

    See `MicroRunner` for parameter descriptions.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        serial_path: Union[str, Path],
        device_path: Union[str, Path],
    ):
        super().__init__(
            directory=directory,
            serial_path=serial_path,
            device_path=device_path,
            binary_name="nengoedge_project.bin",
        )


class NordicRunner(MicroRunner):
    """
    Run a model exported from NengoEdge on the Nordic board.

    See `MicroRunner` for parameter descriptions.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        serial_path: Union[str, Path],
        device_path: Union[str, Path],
    ):
        super().__init__(
            directory=directory,
            serial_path=serial_path,
            device_path=device_path,
            binary_name="nengoedge_project.hex",
        )
