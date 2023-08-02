"""Interface for running an exported NengoEdge model in SavedModel format."""

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import tensorflow as tf


class SavedModelRunner:
    """Run a model exported in TensorFlow's SavedModel format."""

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)

        self.model = tf.keras.saving.load_model(directory, compile=False)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset the internal state of the model to initial conditions."""

        self.state: Optional[List[tf.Tensor]] = None

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
        n_states = len(self.model.inputs) - len(inputs)
        batch_size = inputs[0].shape[0]

        model_inputs = tf.nest.flatten(inputs)

        if self.state is None:
            self.state = [
                tf.zeros(
                    [batch_size] + [0 if s is None else s for s in input.shape[1:]]
                )
                for input in self.model.inputs[len(inputs) :]
            ]

        outputs = tf.nest.flatten(self.model(model_inputs + self.state))

        # Update saved state
        self.state = outputs[len(outputs) - n_states :]

        result = [x.numpy() for x in outputs[: len(outputs) - n_states]]
        if len(result) == 1:
            return result[0]
        return result
