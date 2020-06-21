import skimage.color
import tensorflow as tf
import numpy as np


def rgb_to_cielab(images: np.ndarray) -> np.ndarray:
    """
    Convert RGB to CIE Lab colorspace.
    This method works with single images and batches of images.

    Args:
        images: 3 or 4 tensor where the last three dimensions are (H, W, 3) representing RGB images.
            If float then the values are expected in [0, 1], if integer then the values are expected in [0, 255].

    Returns:
        Images in CIE Lab space as floating point values with ranges [0, 1] for L and [-1, 1) for a and b.
    """

    # Rescale intensity to [0, 1] and a,b to [-1, 1). Note that a,b are non-linear!
    return skimage.color.rgb2lab(images) / (100.0, 128.0, 128.0)


@tf.function
def map_colorspace(images: tf.Tensor) -> tf.Tensor:
    """
    TensorFlow graph function which converts images from RGB to CIE Lab colorspace.
    This is essentially a wrapper for rgb_to_cielab to be usable in a graph.

    Args:
        images: 3 or 4 tensor. See rgb_to_cielab for detailed information.

    Returns:
        Converted images, see rgb_to_cielab for detailed information.
    """
    [images_lab, ] = tf.py_function(rgb_to_cielab, [images], [tf.float32])

    # Make sure shape information is correct after py_function call
    images_lab.set_shape(images.get_shape())

    return images_lab
