# pylint: disable=missing-docstring

import struct
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pytest
import serial

from nengo_edge.micro_runner import DiscoRunner, MicroRunner, NordicRunner


class MockSerial:
    def __init__(self, read_data: List[bytes], write_data: List[bytes]):
        self.read_data = read_data
        self.write_data = write_data
        self.read_count = 0
        self.write_count = 0
        self.timeout = 60

    def read(self, n_bytes: int) -> bytes:
        data = self.read_data[self.read_count]
        self.read_count += 1
        return data

    def write(self, data: bytes) -> None:
        assert data == self.write_data[self.write_count], (
            f"({self.write_count}) "
            f"Expected: {self.write_data[self.write_count]!r} "
            f"Actual: {data!r}"
        )
        self.write_count += 1

    def reset_input_buffer(self) -> None:
        pass

    def reset_output_buffer(self) -> None:
        pass

    def __enter__(self) -> "MockSerial":
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def close(self) -> None:
        pass


def mock_runner(
    tmp_path: Path,
    n_unroll: int,
    monkeypatch: pytest.MonkeyPatch,
    type: str,
) -> MicroRunner:
    device_drive = tmp_path / "micro_device_drive"
    device_drive.mkdir(parents=True, exist_ok=True)

    binary_path = tmp_path / (
        "nengoedge_project.bin" if type == "disco" else "nengoedge_project.hex"
    )
    binary_path.touch()

    def mock_send_data_burst(
        self: MicroRunner,
        serial_port: serial.Serial,
        user_data: np.ndarray,
        state_data: Optional[List[np.ndarray]],
    ) -> None:
        pass

    def mock_receive_data_burst(
        self: MicroRunner,
        serial_port: serial.Serial,
        size: int,
    ) -> np.ndarray:
        if self.model_params["return_sequences"]:
            rxd_data = np.ones((n_unroll * (size // n_unroll)), dtype="float32")
        else:
            rxd_data = np.ones((size), dtype="float32")

        return rxd_data

    monkeypatch.setattr(serial, "Serial", lambda *_, **__: MockSerial([], []))
    monkeypatch.setattr(MicroRunner, "send_data_burst", mock_send_data_burst)
    monkeypatch.setattr(MicroRunner, "receive_data_burst", mock_receive_data_burst)

    runner = (DiscoRunner if type == "disco" else NordicRunner)(
        directory=tmp_path,
        serial_path=Path("/dummy_serial_path"),
        device_path=device_drive,
    )

    return runner


def test_send_data_burst() -> None:
    # device fails to receive ack
    serial_port = MockSerial(
        # device sends request+success, then request+failure, then request+success
        read_data=[b"R", b"S", b"R", b"F", b"R", b"S"],
        # host should send ack, then data (first without state then with state)
        write_data=[
            b"N",
            struct.pack("<f", 0),
            b"T",
            struct.pack("<ff", 1, 2),
            b"T",
            struct.pack("<ff", 1, 2),
        ],
    )

    MicroRunner.send_data_burst(serial_port, np.asarray([0]), None)
    MicroRunner.send_data_burst(serial_port, np.asarray([1]), [np.asarray([2])])
    assert serial_port.read_count == len(serial_port.read_data)
    assert serial_port.write_count == len(serial_port.write_data)

    # device request is lost
    serial_port = MockSerial(
        # timeout on request read, then request+success
        read_data=[b"", b"R", b"S"],
        # host should send failure, then ack+data
        write_data=[b"F", b"N", struct.pack("<f", 0)],
    )
    MicroRunner.send_data_burst(serial_port, np.asarray([0]), None)
    assert serial_port.read_count == len(serial_port.read_data)
    assert serial_port.write_count == len(serial_port.write_data)

    # timeout on send
    serial_port = MockSerial([], [])
    serial_port.timeout = 0
    with pytest.raises(TimeoutError, match="trying to send data to device"):
        MicroRunner.send_data_burst(serial_port, np.asarray([0]), None)


def test_read_data_burst() -> None:
    # ack lost
    serial_port = MockSerial(
        # missing ack, then ack+data
        read_data=[b"", b"T", struct.pack("<f", 1)],
        # host should send req, then failure, then another req, then success
        write_data=[b"R", b"F", b"R", b"S"],
    )
    data = MicroRunner.receive_data_burst(serial_port, 1)
    assert np.array_equal(data, [1])
    assert serial_port.read_count == len(serial_port.read_data)
    assert serial_port.write_count == len(serial_port.write_data)

    # timeout on read
    serial_port = MockSerial([], [])
    serial_port.timeout = 0
    with pytest.raises(TimeoutError, match="trying to receive data from device"):
        MicroRunner.receive_data_burst(serial_port, 1)


def test_runner(
    param_dir: Path,
    rng: np.random.RandomState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_size = 3

    runner = mock_runner(
        tmp_path=param_dir, n_unroll=1, monkeypatch=monkeypatch, type="disco"
    )
    inputs = rng.uniform(-0.5, 0.5, size=(batch_size, 16000))
    with runner:
        # check that batched and non-batched models produce the
        # same output (state is being properly reset between batch items)
        out = []
        for i in range(batch_size):
            out.append(runner.run(inputs[i : i + 1]))
        out0 = np.concatenate(out, axis=0)

    with runner:
        out1 = runner.run(inputs)

    assert out0.shape == out1.shape == (batch_size, 49, 10)

    # increased tolerance due to float32 variance
    assert np.allclose(out0, out1, atol=1e-5), np.max(abs(out0 - out1))


def test_runner_errors(
    param_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    rng: np.random.RandomState,
) -> None:
    batch_size = 3

    runner = mock_runner(
        tmp_path=param_dir, n_unroll=1, monkeypatch=monkeypatch, type="nordic"
    )

    inputs = rng.uniform(-0.5, 0.5, size=(batch_size, 10))

    # check sample rate too low
    runner.preprocessing["sample_rate"] = 10
    with runner:
        with pytest.raises(ValueError, match="sample_rate >> 100"):
            runner.run(inputs)

    runner.preprocessing["sample_rate"] = 16000

    # check n_steps % n_unroll != 0
    runner.model_params["input_shape"][0] = 3

    inputs = rng.uniform(-0.5, 0.5, size=(16000,))
    with runner:
        with pytest.raises(ValueError, match="Inputs must have ndim=2"):
            runner.run(inputs)

        inputs = np.reshape(inputs, (1, 16000))
        with pytest.raises(ValueError, match="cannot be evenly divided by unroll"):
            runner.run(inputs)
