import os
import random

import numpy as np
import tensorflow as tf

DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
DEFAULT_DATA_DIR = os.path.join(DEFAULT_BASE_DIR, 'data')
DEFAULT_LOG_DIR = os.path.join(DEFAULT_BASE_DIR, 'log')


def fix_seeds(seed):
    """
    Fixes seeds for all known instances/libraries.
    Args:
        seed: Seed to use
    """

    # Python object hashes
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Python random library
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Tensorflow
    # See the method's documentation on how this impacts tensorflow randomness
    tf.random.set_seed(seed)


@tf.function
def pad_to_stride(inputs: tf.Tensor, target_stride: int, mode: str = 'REFLECT') -> tf.Tensor:
    """
    Pads the given 4D tensor such that its spatial dimensions become a multiple of the given target_stride.
    Specifically, this applies the smallest padding to axes at index 1 and 2 of inputs
    using the given mode such that the resulting dimensions become a multiple of target_stride.

    This method is generally used to ensure input images can be used as inputs
    to a CNN with a fixed output stride.

    Args:
        inputs: 4D tensor where axes at index 1 and 2 are assumed to be spatial (e.g. in NHWC data format).
        target_stride: Axes are padded to a multiple of this number.
        mode: Padding mode to apply; see `tf.pad` for possible options.

    Returns:
        Padded inputs
    """

    # Calculate total amount to be padded
    missing_y = target_stride - (inputs.shape[1] % target_stride)
    missing_x = target_stride - (inputs.shape[2] % target_stride)

    # Calculate paddings
    # In asymmetric cases the larger padding happens after the features
    paddings = (
        (0, 0),  # Batch
        (missing_y // 2, tf.math.ceil(missing_y / 2)),  # Height
        (missing_x // 2, tf.math.ceil(missing_x / 2)),  # Width
        (0, 0)  # Channels
    )

    return tf.pad(inputs, paddings, mode=mode)
