"""Interface for running an exported NengoEdge model in TFLite format."""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf

from nengo_edge import config


class TFLiteRunner:
    """
    Run an exported TFLite model.

    Parameters
    ----------
    directory : Union[str, Path]
        Path to the directory containing the exported model files.
    """

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)
        self.model_params, self.preprocessing = config.load_params(self.directory)

        # Build interpreters
        model_path = self.directory / "model.tflite"
        if not model_path.exists():
            raise FileNotFoundError(f"Could not find model file ({model_path})")
        self.model_interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.model_interpreter.allocate_tensors()

        feature_path = self.directory / "feature_extractor.tflite"
        if not feature_path.exists():
            warnings.warn(
                f"Could not find feature extractor model file ({feature_path}).\n"
                f"Using the tflite runner without a compiled feature extractor is "
                f"an experimental feature and may lead to poor model performance."
            )
            self.feature_interpreter = None
        else:
            self.feature_interpreter = tf.lite.Interpreter(model_path=str(feature_path))
            self.feature_interpreter.allocate_tensors()

        self.reset_state()

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset the internal state of the model to initial conditions."""
        assert self.model_interpreter is not None
        self.model_state = [
            np.zeros([batch_size] + x["shape"].tolist()[1:], dtype=x["dtype"])
            for x in self.model_interpreter.get_input_details()[1:]
        ]

        if self.feature_interpreter is not None:
            self.feature_state = [
                np.zeros(
                    [
                        # initialize dynamic state sizes to 0 to avoid startup artifacts
                        batch_size if i == 0 else (0 if sig == -1 else val)
                        for i, (sig, val) in enumerate(
                            zip(x["shape_signature"], x["shape"])
                        )
                    ],
                    dtype=x["dtype"],
                )
                for x in self.feature_interpreter.get_input_details()[1:]
            ]

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the model on the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values (should have shape ``(batch_size, input_steps)``).

        Returns
        -------
        outputs : ``np.ndarray``
            Model output values (with shape ``(batch_size, output_d)`` if
            ``self.model_params['return_sequences']=False``
            else ``(batch_size, output_steps, output_d)``).
        """
        if self.feature_interpreter is not None:
            inputs, self.feature_state = self._run_feature_extractor(inputs)

        outputs, self.model_state = self._run_model(inputs)

        return outputs

    def _resize_inputs(
        self, interpreter: tf.lite.Interpreter, input_vals: List[np.ndarray]
    ) -> None:
        reallocate = False
        for s0, s1 in zip(input_vals, interpreter.get_input_details()):
            if s0.shape != tuple(s1["shape"]):
                # this will attempt to resize any dynamic dimensions (e.g. batch
                # size), and will raise an error if there is a mismatch on
                # static dimensions
                interpreter.resize_tensor_input(s1["index"], s0.shape, strict=True)
                reallocate = True
        if reallocate:
            interpreter.allocate_tensors()

    def _run_feature_extractor(
        self, inputs: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run the feature extractor on the given inputs."""
        assert self.feature_interpreter is not None

        self._resize_inputs(self.feature_interpreter, [inputs] + self.feature_state)

        for inp, val in zip(
            self.feature_interpreter.get_input_details(), [inputs] + self.feature_state
        ):
            self.feature_interpreter.set_tensor(inp["index"], val.astype(np.float32))

        self.feature_interpreter.invoke()

        outputs = self.feature_interpreter.get_tensor(
            self.feature_interpreter.get_output_details()[0]["index"]
        )
        feature_state = [
            self.feature_interpreter.get_tensor(out["index"])
            for out in self.feature_interpreter.get_output_details()[1:]
        ]

        return outputs, feature_state

    def _run_model(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run the main model on the given inputs."""

        n_unroll = self.model_interpreter.get_input_details()[0]["shape"][1]
        n_steps = inputs.shape[1]
        if n_steps % n_unroll != 0:
            raise ValueError(
                f"Number of timesteps ({n_steps}) cannot be evenly divided by unroll "
                f"size ({n_unroll})"
            )

        self._resize_inputs(
            self.model_interpreter, [inputs[:, :n_unroll]] + self.model_state
        )
        input_details = self.model_interpreter.get_input_details()
        output_details = self.model_interpreter.get_output_details()

        input_tensors = [
            self.model_interpreter.tensor(details["index"]) for details in input_details
        ]
        output_tensors = [
            self.model_interpreter.tensor(details["index"])
            for details in output_details
        ]

        # quantize all the inputs ahead of time (for efficiency)
        inputs = self.quantize(inputs, input_details[0])

        state_input = self.model_state
        if self.model_params["return_sequences"]:
            output_sequences = []

        for step in range(0, n_steps, n_unroll):
            # set input for current block of unrolled steps
            input_tensors[0]()[:] = inputs[:, step : step + n_unroll]
            for inp, fp32_data, details in zip(
                input_tensors[1:], state_input, input_details[1:]
            ):
                inp()[:] = self.quantize(fp32_data, details)

            # run interpreter
            self.model_interpreter.invoke()

            # get state input for next block of steps
            state_input = [
                self.dequantize(out(), details)
                for out, details in zip(output_tensors[1:], output_details[1:])
            ]

            if self.model_params["return_sequences"]:
                # Append all outputs from the past `unroll` timesteps
                output_sequences.append(output_tensors[0]().copy())

        if self.model_params["return_sequences"]:
            # concatenate outputs from each unroll block
            # note: because the model was built with return_sequences=True, this
            # represents the output from all the timesteps
            outputs = np.concatenate(output_sequences, axis=1)
        else:
            # only use the output from last unroll block
            # note: because the model was built with return_sequences=False, the
            # output from the last unroll block is actually the output from the
            # last timestep
            outputs = output_tensors[0]().copy()

        # dequantize outputs (state_input was already dequantized in the loop above)
        outputs = self.dequantize(outputs, output_details[0])

        return outputs, state_input

    @staticmethod
    def quantize(val: np.ndarray, details: Dict[str, Any]) -> np.ndarray:
        """Quantize the given value based on quantization parameters."""
        n_scales = len(details["quantization_parameters"]["scales"])
        if n_scales == 0:
            # val is not quantized
            return np.array(val)

        assert n_scales == 1
        val = np.asarray(val)
        scaling_factor = details["quantization_parameters"]["scales"][0]
        zero_point = details["quantization_parameters"]["zero_points"][0]
        iinfo = np.iinfo(details["dtype"])
        q = np.round((val / scaling_factor) + zero_point)
        return np.clip(q, iinfo.min, iinfo.max).astype(details["dtype"])

    @staticmethod
    def dequantize(val: np.ndarray, details: Dict[str, Any]) -> np.ndarray:
        """Dequantize the given value based on quantization parameters."""
        n_scales = len(details["quantization_parameters"]["scales"])
        if n_scales == 0:
            # val is not quantized
            return np.array(val)

        assert n_scales == 1
        scaling_factor = details["quantization_parameters"]["scales"][0]
        zero_point = details["quantization_parameters"]["zero_points"][0]
        return (val.astype(np.float32) - zero_point) * scaling_factor
