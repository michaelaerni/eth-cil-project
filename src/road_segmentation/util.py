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
    TODO: Documentation
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
