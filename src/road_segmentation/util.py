import datetime
import logging
import os
import random

import numpy as np
import tensorflow as tf

DEFAULT_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 2)))
DEFAULT_DATA_DIR = os.path.join(DEFAULT_BASE_DIR, 'data')
DEFAULT_LOG_DIR = os.path.join(DEFAULT_BASE_DIR, 'log')

_LOG_FORMAT = '%(asctime)s  %(levelname)s [%(name)s]: %(message)s'


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def setup_experiment(log_directory: str, experiment_tag: str) -> str:
    directory_name = f'{experiment_tag}_{datetime.datetime.now():%y%m%d-%H%M%S}'
    experiment_dir = os.path.join(log_directory, directory_name)
    os.makedirs(experiment_dir, exist_ok=False)
    return experiment_dir


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
