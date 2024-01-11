"""Interface for running an exported NengoEdge model in SavedModel format."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from nengo_edge import config

try:
    import tensorflow as tf
    from tensorflow_text import SentencepieceTokenizer

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
            self.tokenizer = self.get_tokenizer(self.postprocessing)
            if self.tokenizer is None:
                warnings.warn(
                    "No tokenizer model found, cannot decode ASR outputs. "
                    "Consider re-downloading ASR run artifacts."
                )
        elif self.model_params["type"] == "nlp":
            self.tokenizer = self.get_tokenizer(self.preprocessing)
            if self.tokenizer is None:
                raise ValueError(
                    "No tokenizer model found and is required to process string inputs."
                    "Please re-download the NLP artifacts."
                )

    def get_tokenizer(
        self, params: Dict[str, str]
    ) -> Optional["SentencepieceTokenizer"]:
        """
        Load a ``tensorflow_text.SentencepieceTokenizer``.

        Returns None if no ``tokenizer_file`` exists in the load directory.

        Parameters
        ----------
        params: Dict[str, str]
            Configuration dictionary containing the name of the
            ``tokenizer_file`` found in the save directory.

        Returns
        -------
        tokenizer: ``tensorflow_text.SentencepieceTokenizer``
            A sentencepiece tokenizer with a trained vocabulary.
        """
        if (
            "tokenizer_file" in params
            and (self.directory / params["tokenizer_file"]).exists()
        ):
            return SentencepieceTokenizer(
                (self.directory / params["tokenizer_file"]).read_bytes()
            )
        return None

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

        model_inputs = tf.nest.flatten(masked_inputs)

        output = tf.nest.flatten(self.model(model_inputs))[0]

        return ragged.from_masked(output).numpy()

    def run(self, inputs: Union[np.ndarray, List[str]]) -> np.ndarray:
        """
        Run the model on the given inputs.

        This function applies model specific actions to inputs or outputs.
        For example, ASR model outputs are automatically detokenized to construct
        a string prediction. For NLP models, the list of strings is
        automatically processed into integer tokens from a saved
        vocabulary.

        Parameters
        ----------
        inputs : Union[np.ndarray, List[str]]
            For audio based models, such as KWS and ASR, model input values
            should be stored in a numpy array with a shape of
            ``(batch_size, input_steps)``.
            For text based NLP models, input strings should be stored in a list with
            a length equal to ``batch_size``.

        Returns
        -------
        outputs : np.ndarray
            Model output values with shape ``(batch_size, output_steps, output_d)``
            for KWS, ``(batch_size)`` for ASR and ``(batch_size, output_d)`` for NLP.
        """

        if isinstance(inputs[0], str):
            # Process string inputs
            assert self.model_params["type"] == "nlp"
            assert self.tokenizer is not None
            inputs = self.tokenizer.tokenize([s.lower() for s in inputs]).numpy()

        assert isinstance(inputs, np.ndarray)
        if self.model_params["type"] == "kws" and inputs.dtype == object:
            raise NotImplementedError("KWS models do not support ragged inputs")

        outputs = self._run_model(inputs)

        # Detokenize asr outputs
        if self.model_params["type"] == "asr" and self.tokenizer is not None:
            # Apply CTC decoding
            outputs = tf.keras.backend.ctc_decode(
                tf.nn.softmax(ragged.to_masked(outputs)),
                input_length=[len(x) for x in outputs],
                greedy=True,
            )[0][0]
            outputs = tf.cast(outputs, "int32")

            # Filter out blank/unknown tokens
            outputs = tf.ragged.boolean_mask(outputs, outputs > 0)

            # Convert tokens back to text
            outputs = np.asarray(
                [s.decode("utf-8") for s in self.tokenizer.detokenize(outputs).numpy()]
            )

        return outputs
