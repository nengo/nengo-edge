"""Interface for running an exported NengoEdge model in SavedModel format."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from nengo_edge import config


class SavedModelRunner:
    """Run a model exported in TensorFlow's SavedModel format."""

    def __init__(self, directory: Union[str, Path]):
        self.directory = Path(directory)

        self.model = tf.keras.saving.load_model(directory, compile=False)

        self.model_params, self.preprocessing = config.load_params(self.directory)

        self.reset_state()

    def reset_state(self) -> None:
        """Reset the internal state of the model to initial conditions."""

        self.state: Optional[List[tf.Tensor]] = None

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
            the model was built to return only the final time step,
            else ``(batch_size, output_steps, output_d)``).
        """

        inputs = tf.cast(inputs, "float32")
        batch_size = inputs.shape[0]

        model_inputs = tf.nest.flatten(inputs)

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

        return outputs[0].numpy()
