"""Tools for working with NengoEdge."""

import numpy as np
from typing_extensions import Protocol

from nengo_edge.version import copyright as __copyright__
from nengo_edge.version import version as __version__


class Runner(Protocol):
    """Run a model exported by NengoEdge."""

    def reset_state(self) -> None:
        """Reset the internal state of the model to initial conditions."""

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """
        Run the model on the given inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Model input values.

        Returns
        -------
        outputs : ``np.ndarray``
            Model output values.
        """
