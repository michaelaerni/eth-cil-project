import typing

import numpy as np
import skimage.color
import tensorflow as tf


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


@tf.function
def random_grayscale(image: tf.Tensor, probability: typing.Union[float, tf.Tensor]) -> tf.Tensor:
    """
    Converts the given image to grayscale with the given probability.

    Args:
        image: Image to randomly convert.
        probability: Probability of grayscale conversion in range [0, 1].

    Returns:
        Image converted to grayscale with the given probability, unchanged with 1 - probability.
    """
    # Determine whether to convert or not
    do_convert = tf.random.uniform(shape=[], dtype=tf.float32) < probability
    output_image = tf.cond(do_convert, lambda: tf.repeat(tf.image.rgb_to_grayscale(image), 3, axis=-1), lambda: image)

    # Must set shape manually since it cannot be inferred from tf.cond
    output_image.set_shape(image.shape)

    return output_image


def random_color_jitter(
        image: tf.Tensor,
        brightness_jitter: float,
        contrast_jitter: float,
        saturation_jitter: float,
        hue_jitter: float
) -> tf.Tensor:
    """
    Applies random brightness, contrast, saturation and hue changes to the given image.

    Args:
        image: Image to augment as a tensor of rank at least 3 where the last 3 dimensions are (H, W, C).
            If a floating point image then values should be in [0, 1].
        brightness_jitter: How much to jitter brightness. Should be in [0, 1].
        contrast_jitter: How much to jitter contrast. Should be in [0, 1].
        saturation_jitter: How much to jitter saturation. Should be in [0, 1].
        hue_jitter: How much to jitter hue. Should be in [0, 0.5].

    Returns:
        Image with random jitter applied.
    """

    image = tf.image.random_brightness(image, brightness_jitter)

    image = tf.image.random_contrast(image, lower=max(0.0, 1.0 - contrast_jitter), upper=1 + contrast_jitter)
    image = tf.image.random_saturation(image, lower=max(0.0, 1.0 - saturation_jitter), upper=1 + saturation_jitter)
    image = tf.image.random_hue(image, hue_jitter)

    return image
