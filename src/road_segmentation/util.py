import abc
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


class SplitBatchNormalization(tf.keras.layers.Layer):
    def __init__(
            self,
            gamma_initializer: str = None,
            partitions: int = 4,
            **kwargs
    ):
        super(SplitBatchNormalization, self).__init__(**kwargs)

        self.partitions = partitions
        self.partition_size = None

        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization(gamma_initializer=gamma_initializer)
            for _ in range(partitions)
        ]

    def call(self, inputs, **kwargs):
        # Directly returning batch norm applied to inputs yelds a different result than replacing this layer with
        # a standard batch norm layer. The reason of this is unclear.
        # return self.batch_norm_layers[0](inputs)
        batch_size = tf.shape(inputs)[0]

        separate_normalized = []
        # if self.partition_size is None:
        self.partition_size = tf.cast(tf.math.ceil(batch_size / self.partitions), dtype=tf.int32)

        for partition_idx, batch_norm in enumerate(self.batch_norm_layers):
            partition_start = partition_idx * self.partition_size
            partition_end = partition_start + self.partition_size

            normalized = batch_norm(inputs[partition_start:partition_end]),
            separate_normalized.append(normalized)

        # It fails if gather is not called for some reason.
        normalized = tf.gather(tf.concat(separate_normalized, axis=0), tf.range(batch_size), axis=0)
        return normalized


class ShuffleSplitBatchNormalization(tf.keras.layers.Layer):
    def __init__(
            self,
            gamma_initializer: str = None,
            partitions: int = 4,
            **kwargs
    ):
        super(ShuffleSplitBatchNormalization, self).__init__(**kwargs)

        self.partitions = partitions
        self.partition_size = None

        self.batch_norm_layers = [
            tf.keras.layers.BatchNormalization(gamma_initializer=gamma_initializer)
            for _ in range(partitions)
        ]

    def call(self, inputs, **kwargs):
        batch_size = tf.shape(inputs)[0]
        permuted_indices = tf.random.shuffle(tf.range(batch_size))
        permuted_indices_inv = tf.math.invert_permutation(permuted_indices)

        shuffled = tf.gather(inputs, permuted_indices, axis=0)

        separate_normalized = []
        self.partition_size = tf.cast(tf.math.ceil(batch_size / self.partitions), dtype=tf.int32)

        for partition_idx, batch_norm in enumerate(self.batch_norm_layers):
            partition_start = partition_idx * self.partition_size
            partition_end = partition_start + self.partition_size

            normalized = batch_norm(shuffled[partition_start:partition_end])
            separate_normalized.append(normalized)

        shuffled_normalized = tf.concat(separate_normalized, axis=0)
        normalized = tf.gather(shuffled_normalized, permuted_indices_inv, axis=0)
        return normalized


class NormalizationBuilder(metaclass=abc.ABCMeta):
    """
    Callable builder which creates normalization layers.
    """

    @abc.abstractmethod
    def __call__(self, zero_init: bool = False) -> tf.keras.layers.Layer:
        """
        Build a new normalization layer.
        The resulting normalization layer has a learnable bias term included.

        Args:
            zero_init: If true then the normalization layer will be zero-initialized. If false, defaults apply.

        Returns:
            New normalization layer.
        """
        pass


class BatchNormalizationBuilder(NormalizationBuilder):
    """
    Normalization layer builder for tf.keras.layers.BatchNormalization layers.
    """

    def __call__(self, zero_init: bool = False) -> tf.keras.layers.Layer:
        gamma_initializer = 'zeros' if zero_init else 'ones'
        return tf.keras.layers.BatchNormalization(gamma_initializer=gamma_initializer)


class LayerNormalizationBuilder(NormalizationBuilder):
    """
    Normalization layer builder for tf.keras.layers.LayerNormalization layers.
    """

    def __call__(self, zero_init: bool = False) -> tf.keras.layers.Layer:
        gamma_initializer = 'zeros' if zero_init else 'ones'
        return tf.keras.layers.LayerNormalization(gamma_initializer=gamma_initializer)


class SplitBatchNormalizationBuilder(NormalizationBuilder):
    """
    Normalization layer builder for custom split batch normalization
    """

    def __call__(self, zero_init: bool = False) -> tf.keras.layers.Layer:
        gamma_initializer = 'zeros' if zero_init else 'ones'
        return SplitBatchNormalization(gamma_initializer=gamma_initializer)


class ShuffleSplitBatchNormalizationBuilder(NormalizationBuilder):
    """
    Normalization layer builder for custom shuffled and split batch normalization
    """

    def __call__(self, zero_init: bool = False) -> tf.keras.layers.Layer:
        gamma_initializer = 'zeros' if zero_init else 'ones'
        return ShuffleSplitBatchNormalization(gamma_initializer=gamma_initializer)
