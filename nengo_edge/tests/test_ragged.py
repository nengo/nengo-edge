# pylint: disable=missing-docstring

from typing import Iterable

import numpy as np
import pytest
import tensorflow as tf

from nengo_edge import ragged


@pytest.mark.parametrize(
    "val",
    (
        [[0, 1, 2], [3, 4]],
        np.array([[0, 1, 2], [3, 4]], dtype=object),
        tf.zeros((2, 3)),
        tf.ragged.constant([[0, 1, 2], [3, 4]]),
    ),
)
def test_to_from_masked(val: Iterable) -> None:
    masked = ragged.to_masked(val)
    assert np.array_equal(
        tf.math.count_nonzero(masked._keras_mask, axis=-1), [len(x) for x in val]
    )

    unmasked = ragged.from_masked(masked)
    assert np.array_equal([len(x) for x in unmasked], [len(x) for x in val])
