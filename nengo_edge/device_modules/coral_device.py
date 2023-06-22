"""On device runner for the Coral dev board."""

import argparse
import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from nengo_edge.device_modules.np_mfcc import LogMelFeatureExtractor


class CoralDeviceRunner:  # pragma: no cover (runs on-device)
    """
    Run a compiled model on the Coral board.

    Parameters
    ----------
    directory : Path
        Directory containing the edgetpu TFLite model file that will be executed.
    return_sequences: bool
        Flag to return all outputs
    """

    def __init__(
        self,
        directory: Union[str, Path],
        return_sequences: bool,
    ):
        self.directory = Path(directory)
        self.return_sequences = return_sequences

        # Build interpreter
        # pylint: disable=import-outside-toplevel
        import tflite_runtime.interpreter as tflite

        model_files = list(self.directory.glob("*_edgetpu.tflite"))
        if len(model_files) == 0:
            raise FileNotFoundError(f"No model files in {self.directory}")
        elif len(model_files) > 1:
            raise ValueError(f"Found multiple model files {model_files}")

        self.interpreter = tflite.Interpreter(
            model_path=str(model_files[0]),
            experimental_delegates=[
                tflite.load_delegate(
                    {
                        "Linux": "libedgetpu.so.1",
                        "Darwin": "libedgetpu.1.dylib",
                        "Windows": "edgetpu.dll",
                    }[platform.system()],
                )
            ],
        )
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.interpreter.allocate_tensors()

        self.n_unroll = self.input_details[0]["shape"][1]

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the model on the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values (should have shape ``(batch_size, n_steps, input_d)``.

        Returns
        -------
            Model output values (with shape ``(batch_size, output_d)`` if
            ``self.return_sequences=False``  else
            ``(batch_size, output_steps, output_d)``).
        """

        assert inputs.shape[0] == 1
        n_steps = inputs.shape[1]
        assert n_steps % self.n_unroll == 0

        input_tensors = [
            self.interpreter.tensor(details["index"]) for details in self.input_details
        ]
        output_tensors = [
            self.interpreter.tensor(details["index"]) for details in self.output_details
        ]

        # quantize all the inputs ahead of time (for efficiency)
        inputs = self.quantize(inputs, self.input_details[0])

        state_input = [
            np.zeros(x["shape"], dtype=x["dtype"]) for x in self.input_details[1:]
        ]
        output_sequences = []
        for step in range(0, n_steps, self.n_unroll):
            # set input for current block of unrolled steps
            input_tensors[0]()[:] = inputs[:, step : step + self.n_unroll]
            for inp, fp32_data, details in zip(
                input_tensors[1:], state_input, self.input_details[1:]
            ):
                inp()[:] = self.quantize(fp32_data, details)

            # run interpreter
            self.interpreter.invoke()

            # get state input for next block of steps
            state_input = [
                self.dequantize(out(), details)
                for out, details in zip(output_tensors[1:], self.output_details[1:])
            ]

            if self.return_sequences:
                # Append all outputs from the past `unroll` timesteps
                output_sequences.append(output_tensors[0]().copy())

        outputs = (
            # concatenate outputs from each unroll block
            np.concatenate(output_sequences, axis=1)
            if self.return_sequences
            # only use the output from last unroll block
            else output_tensors[0]().copy()
        )

        # dequantize outputs
        outputs = self.dequantize(outputs, self.output_details[0])

        return outputs

    @staticmethod
    def quantize(val: np.ndarray, details: Dict[str, np.ndarray]) -> np.ndarray:
        """Quantize the given value based on quantization parameters."""

        val = np.asarray(val)
        assert len(details["quantization_parameters"]["scales"]) == 1
        assert len(details["quantization_parameters"]["zero_points"]) == 1
        scaling_factor = details["quantization_parameters"]["scales"][0]
        zero_point = details["quantization_parameters"]["zero_points"][0]
        iinfo = np.iinfo(details["dtype"])
        q = np.round((val / scaling_factor) + zero_point)
        return np.clip(q, iinfo.min, iinfo.max).astype(details["dtype"])

    @staticmethod
    def dequantize(val: np.ndarray, details: Dict[str, np.ndarray]) -> np.ndarray:
        """Dequantize the given value based on quantization parameters."""

        assert len(details["quantization_parameters"]["scales"]) == 1
        assert len(details["quantization_parameters"]["zero_points"]) == 1
        scaling_factor = details["quantization_parameters"]["scales"][0]
        zero_point = details["quantization_parameters"]["zero_points"][0]
        return (val.astype(np.float32) - zero_point) * scaling_factor


def _build_feature_extractor(
    args: argparse.Namespace,
) -> "LogMelFeatureExtractor":  # pragma: no cover (runs on-device)
    # pylint: disable=import-outside-toplevel

    from np_mfcc import LogMelFeatureExtractor  # type: ignore

    with open(Path(args.directory) / "parameters.json", "r", encoding="utf-8") as f:
        parameters = json.load(f)

    options = parameters["preprocessing"]
    feature_extractor = LogMelFeatureExtractor(**options)

    return feature_extractor


def _run_model(
    feature_extractor: "LogMelFeatureExtractor",
    runner: CoralDeviceRunner,
    inputs: np.ndarray,
) -> np.ndarray:
    inputs = np.concatenate(
        [feature_extractor(inputs[i : i + 1]) for i in range(inputs.shape[0])],
        axis=0,
    )

    # coral board only supports batch size of 1, so we have to manually iterate
    # over batches
    out_samples = []
    for i in range(inputs.shape[0]):
        out_samples.append(runner.run(inputs=inputs[i : i + 1]))
    out = np.concatenate(out_samples, axis=0)

    return out


def cli() -> None:
    """Command line interface for running on device."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        required=True,
        help=(
            "Path to build artifacts on coral board"
            " (including the edgetpu model and parameters.json files)"
        ),
    )
    parser.add_argument(
        "--return-sequences",
        action="store_true",
        help="Return all outputs generated by the model",
    )
    args = parser.parse_args()

    # set up feature extractor
    feature_extractor = _build_feature_extractor(args)

    # build model/interpreter
    runner = CoralDeviceRunner(
        directory=Path(args.directory), return_sequences=args.return_sequences
    )

    # load inputs from file
    with np.load(Path(args.directory) / "inputs.npz", allow_pickle=False) as data:
        inputs = data["inputs"]

    # run model
    out = _run_model(feature_extractor, runner, inputs)

    # save outputs to file
    np.save(Path(args.directory) / "outputs.npy", out, allow_pickle=False)


if __name__ == "__main__":  # pragma: no cover (runs on-device)
    cli()
