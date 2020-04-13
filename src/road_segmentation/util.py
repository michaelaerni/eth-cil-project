import datetime
import logging
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
