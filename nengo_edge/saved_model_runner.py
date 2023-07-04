"""Interface for running an exported NengoEdge model in SavedModel format."""

from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np
import tensorflow as tf


class SavedModelRunner:
    """Run a model exported in TensorFlow's SavedModel format."""

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)

        self.model = tf.saved_model.load(str(self.directory)).signatures[
            "serving_default"
        ]

        # The saved model takes inputs and returns outputs organized by name. But in
        # general we don't know what those names will be, because they can depend on
        # what other models have been loaded in the TensorFlow graph. However, when
        # creating the models in NengoEdge we ensure that the inputs/outputs will be
        # ordered alphabetically, so we can use that to recover the correct
        # input/output order here.
        self.input_names = sorted(self.model.structured_input_signature[1])
        self.output_names = sorted(self.model.structured_outputs)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset the internal state of the model to initial conditions."""

        self.state: Dict[str, tf.Tensor] = {}

    def run(
        self, inputs: Union[np.ndarray, Sequence[np.ndarray]]
    ) -> Union[np.ndarray, Sequence[np.ndarray]]:
        """
        Run the model on the given inputs.

        Parameters
        ----------
        inputs : Union[np.ndarray, Sequence[np.ndarray]]
            Model input values (should have shape ``(batch_size, input_steps)``).

        Returns
        -------
        outputs : Union[np.ndarray, Sequence[np.ndarray]]
            Model output values (with shape ``(batch_size, output_d)`` if
            the model was built to return only the final time step,
            else ``(batch_size, output_steps, output_d)``).
        """

        inputs = [tf.cast(x, "float32") for x in tf.nest.flatten(inputs)]
        n_states = len(self.model.structured_input_signature[1]) - len(inputs)
        batch_size = inputs[0].shape[0]

        kwargs = {}
        for name, sig in self.model.structured_input_signature[1].items():
            if name in self.input_names[: len(inputs)]:
                kwargs[name] = inputs[self.input_names.index(name)]
            else:
                if name not in self.state:
                    self.state[name] = tf.zeros(
                        [batch_size] + [0 if s is None else s for s in sig.shape[1:]]
                    )
                kwargs[name] = self.state[name]

        outputs = self.model(**kwargs)

        # Update saved state
        for input_name, output_name in zip(
            self.input_names[-n_states:], self.output_names[-n_states:]
        ):
            self.state[input_name] = outputs[output_name]

        result = [
            outputs[n].numpy()
            for n in self.output_names[: len(self.output_names) - n_states]
        ]
        if len(result) == 1:
            return result[0]
        return result
