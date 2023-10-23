"""Interface for running an exported NengoEdge model in SavedModel format."""

import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from nengo_edge import config

try:
    import tensorflow as tf
    from tensorflow_text import FastSentencepieceTokenizer

    from nengo_edge import ragged  # pylint: disable=ungrouped-imports
except ImportError:  # pragma: no cover
    warnings.warn("TensorFlow is not installed; cannot use SavedModelRunner.")


class SavedModelRunner:
    """Run a model exported in TensorFlow's SavedModel format."""

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)

        self.model = tf.keras.saving.load_model(directory, compile=False)

        self.model_params, self.preprocessing, self.postprocessing = config.load_params(
            self.directory
        )

        if self.model_params["type"] == "asr":
            if (
                "tokenizer_file" in self.postprocessing
                and (self.directory / self.postprocessing["tokenizer_file"]).exists()
            ):
                self.tokenizer = FastSentencepieceTokenizer(
                    (
                        self.directory / self.postprocessing["tokenizer_file"]
                    ).read_bytes()
                )
            else:
                self.tokenizer = None
                warnings.warn(
                    "No tokenizer model found, cannot decode ASR outputs. "
                    "Consider re-downloading ASR run artifacts."
                )

        self.reset_state()

    def reset_state(self) -> None:
        """Reset the internal state of the model to initial conditions."""

        self.state: Optional[List[tf.Tensor]] = None

    def _run_model(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the main model logic on the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values (should have shape ``(batch_size, input_steps)``).

        Returns
        -------
        outputs : NDarray
            Model output values in a ragged tensor.
        """

        ragged_inputs = ragged.to_tf(inputs)
        ragged_inputs = tf.cast(ragged_inputs, "float32")
        masked_inputs = ragged.to_masked(ragged_inputs)

        batch_size = masked_inputs.shape[0]
        model_inputs = tf.nest.flatten(masked_inputs)

        if self.state is None:
            self.state = [
                tf.zeros(
                    [batch_size] + [0 if s is None else s for s in state.shape[1:]]
                )
                for state in self.model.inputs[1:]
            ]
        else:
            if not all(s.shape[0] == batch_size for s in self.state):
                raise ValueError(
                    "Input batch size does not match saved state batch size; "
                    "maybe you need to call reset_state()?"
                )

        outputs = tf.nest.flatten(self.model(model_inputs + self.state))

        # Update saved state
        self.state = outputs[1:]

        outputs[0] = ragged.from_masked(outputs[0])

        return outputs[0].numpy()

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the model on the given inputs.

        Also applies model type specific actions to outputs. For example,
        ASR model outputs are automatically detokenized to construct
        a string prediction.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values (should have shape ``(batch_size, input_steps)``).

        Returns
        -------
        outputs : np.ndarray
            Model output values with shape ``(batch_size, output_steps, output_d)``
            for KWS and ``(batch_size)`` for ASR.
        """

        if inputs.dtype == object and self.model_params["type"] == "kws":
            raise NotImplementedError("KWS models do not support ragged inputs")

        outputs = self._run_model(inputs)

        # Detokenize asr outputs using greedy decoding
        if self.model_params["type"] == "asr" and self.tokenizer is not None:
            greedy_outputs = tf.math.argmax(
                ragged.to_masked(outputs), axis=-1, output_type=tf.int32
            )
            greedy_outputs = tf.ragged.boolean_mask(greedy_outputs, greedy_outputs > 0)
            outputs = self.tokenizer.detokenize(greedy_outputs).numpy()

        return outputs
