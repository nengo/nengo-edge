"""Tools for working with ragged arrays in Numpy/TensorFlow."""

from typing import Iterable, Union

import tensorflow as tf


def to_tf(x: Iterable) -> tf.RaggedTensor:
    """Convert inputs to a TensorFlow RaggedTensor."""
    if isinstance(x, tf.RaggedTensor):
        return x
    elif isinstance(x, tf.Tensor):
        return tf.RaggedTensor.from_tensor(x)
    return tf.ragged.constant(x)


def to_masked(x: Iterable) -> tf.Tensor:
    """Convert inputs to a masked TensorFlow Tensor."""
    ragged_x = to_tf(x)
    mask = tf.sequence_mask(ragged_x.row_lengths())
    masked_x = ragged_x.to_tensor()
    masked_x._keras_mask = mask
    return masked_x


def from_masked(x: tf.Tensor) -> Union[tf.Tensor, tf.RaggedTensor]:
    """Convert from a masked Tensor to a RaggedTensor."""

    if tf.reduce_all(getattr(x, "_keras_mask", True)):
        # Mask is all True (or tensor is not masked), so just return as-is
        return x

    return tf.ragged.boolean_mask(x, x._keras_mask)
