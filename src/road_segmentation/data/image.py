import math
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


def random_rotate_and_crop(
        image: tf.Tensor,
        crop_size: typing.Union[int, tf.Tensor]
) -> tf.Tensor:
    """
    Rotates the input image by a random angle, then takes a random crop of given size. The crop location is chosen
    randomly, uniform over all valid crop location. A valid crop location is one where the crop will not contain any
    pixels that had to be filled due to the random rotation.
    Args:
        image: Input image.
        crop_size: Crop size, the output image will be of spatial dimension (crop_size, crop_size).

    Returns:
        The randomly rotated and randomly cropped image.
    """
    input_dimension = image.shape[0]
    angle = tf.random.uniform((), minval=0, maxval=2 * math.pi)

    [image_rotated, ] = tf.py_function(
        lambda image, angle:
            tf.keras.preprocessing.image.apply_affine_transform(
                image.numpy(),
                theta=angle * 180.0 / math.pi,
                channel_axis=2,
                row_axis=0,
                col_axis=1,
                fill_mode='constant',
                cval=0
            ),
        inp=[image, angle],
        Tout=[tf.float32]
    )

    crop_space = _compute_crop_space(angle, input_dimension, crop_size)
    crop_center_unit_coords = tf.random.uniform((4, 1), minval=-1, maxval=1)
    crop_center = tf.matmul(crop_space, crop_center_unit_coords)
    crop_center = tf.sign(crop_center) * tf.floor(tf.abs(crop_center)) + input_dimension // 2

    crop_offset_height = tf.cast(crop_center[0, 0] - crop_size // 2, dtype=tf.int32)
    crop_offset_width = tf.cast(crop_center[1, 0] - crop_size // 2, dtype=tf.int32)

    image_cropped = tf.image.crop_to_bounding_box(
        image=image_rotated,
        offset_height=crop_offset_height,
        offset_width=crop_offset_width,
        target_height=crop_size,
        target_width=crop_size
    )
    return image_cropped


def _compute_crop_space(
        angle: typing.Union[float, tf.Tensor],
        input_dimension: typing.Union[int, tf.Tensor],
        crop_size: typing.Union[int, tf.Tensor]
) -> tf.Tensor:
    """
    Computes a matrix where a subset of the column space describes all valid locations for the center of a crop in an
    image of size `input_dimension` that was rotated by `angle` degrees with a crop size of `crop_size`. When
    multiplying some vector in [-1, 1]^4 with the computed matrix in R^{2x4}, the resulting coordinates will always be
    a valid center for the crop.

    Note: If the crop size does not fit within the valid area of the rotated image, this will lead to a zero matrix.

    Args:
        angle: The angle which the initial image is rotated by.
        input_dimension: The dimensions of the initial image.
        crop_size: The crop size.

    Returns:
        A matrix in R^{2x4}, the columns of which can be used to compute valid center points for cropped image.
        The returned matrix assumes that the (0, 0) coordinate is in the center of the image.
    """
    angle_mod = tf.math.mod(angle, math.pi / 2)

    # Rotation matrix
    rot = [[tf.cos(angle_mod), 0 - tf.sin(angle_mod)], [tf.sin(angle_mod), tf.cos(angle_mod)]]

    input_dimension = input_dimension / 2
    crop_size = crop_size / 2

    top_left = tf.cast([[-input_dimension], [input_dimension]], dtype=tf.float32)
    top_right = tf.cast([[input_dimension], [input_dimension]], dtype=tf.float32)
    bottom_right = tf.cast([[input_dimension], [-input_dimension]], dtype=tf.float32)
    bottom_left = tf.cast([[-input_dimension], [-input_dimension]], dtype=tf.float32)

    top_left_rot = tf.matmul(rot, top_left)
    top_right_rot = tf.matmul(rot, top_right)
    bottom_right_rot = tf.matmul(rot, bottom_right)
    bottom_left_rot = tf.matmul(rot, bottom_left)

    # Compute intersection of the initial square and the rotated square.
    # The convex hull of these intersection points are all pixels which have not
    # been filled by zero values after rotation.
    intersect_top_left = 1 / 2 * (
            (top_left + top_right_rot) -
            (top_left + top_right_rot) / tf.norm(top_left + top_right_rot) *
            tf.norm(top_left - top_right_rot) * tf.math.tan(1 / 2 * angle_mod)
    )
    intersect_top_right = 1 / 2 * (
            (top_right + top_right_rot) -
            (top_right + top_right_rot) / tf.norm(top_right + top_right_rot) *
            tf.norm(top_right - top_right_rot) * tf.math.tan(math.pi / 4 - 1 / 2 * angle_mod)
    )
    intersect_right_top = tf.matmul([[0.0, 1.0], [-1.0, 0.0]], intersect_top_left)
    intersect_right_bottom = tf.matmul([[0.0, 1.0], [-1.0, 0.0]], intersect_top_right)
    intersect_bottom_right = -intersect_top_left
    intersect_bottom_left = -intersect_top_right
    intersect_left_bottom = -intersect_right_top
    intersect_left_top = -intersect_right_bottom

    # Naming: intersect_[side of the not rotated square]_[location on not rotated square]

    # We are interested in the center coordinate of the crop square. The space where
    # the center point may be is smaller than what the convex hull from the previous
    # intersection points allows. By moving each intersection point towards the origin
    # by the half the size of the crop square (on both axes), we restrict the are to
    # the correct smaller area.
    intersect_top_left_inner = intersect_top_left + [[crop_size], [-crop_size]]
    intersect_top_right_inner = intersect_top_right + [[-crop_size], [-crop_size]]
    intersect_right_top_inner = intersect_right_top + [[-crop_size], [-crop_size]]
    intersect_right_bottom_inner = intersect_right_bottom + [[-crop_size], [crop_size]]
    intersect_bottom_right_inner = intersect_bottom_right + [[-crop_size], [crop_size]]
    intersect_bottom_left_inner = intersect_bottom_left + [[crop_size], [crop_size]]
    intersect_left_bottom_inner = intersect_left_bottom + [[crop_size], [crop_size]]
    intersect_left_top_inner = intersect_left_top + [[crop_size], [-crop_size]]

    # The above computation works unless two points cross each other when they are
    # moved inwards. Example: Initially point `intersect_right_top` is above point
    # `intersect_right_bot`. If their distance is less than crop size, then moving
    # the upper point down by half the crop size and the lower point up by half
    # the crop size, will result in `intersect_right_bot` being above
    # `intersect_right_top`. This will lead to the crop square being allowed to
    # include pixels which are outside of the valid area.
    # For the example, this is solved by:
    #   1. Compute intersection point `intersect_bottom` of segments
    #      `(intersect_left_bot, intersect_bot_left)` and
    #      `(intersect_bot_right, intersect_right_bot)`
    #   2. Set `intersect_bot_left = intersect_bottom` and
    #      `intersect_bot_right = intersect_bottom`.
    #
    # Intuitively: the center of the crop square may never cross any of the segments
    # defined by the intersection points. If two segments intersect, then two points
    # one from each segment, can only be reached by crossing one of the two segments.
    # Thus, by moving the points outside to where the segments intersect, the segments
    # no longer are crossed when choosing this new "corner" point.

    if intersect_top_left_inner[0] > intersect_top_right_inner[0]:
        intersect = _segment_intersect(
            intersect_left_top_inner, intersect_top_left_inner,
            intersect_top_right_inner, intersect_right_top_inner
        )
        intersect_top_left_inner = intersect
        intersect_top_right_inner = intersect

    if intersect_right_bottom_inner[1] > intersect_right_top_inner[1]:
        intersect = _segment_intersect(
            intersect_top_right_inner, intersect_right_top_inner,
            intersect_right_bottom_inner, intersect_bottom_right_inner,
        )
        intersect_right_top_inner = intersect
        intersect_right_bottom_inner = intersect

    if intersect_bottom_left_inner[0] > intersect_bottom_right_inner[0]:
        intersect = _segment_intersect(
            intersect_left_bottom_inner, intersect_bottom_left_inner,
            intersect_bottom_right_inner, intersect_right_bottom_inner,
        )
        intersect_bottom_left_inner = intersect
        intersect_bottom_right_inner = intersect

    if intersect_left_bottom_inner[1] > intersect_left_top_inner[1]:
        intersect = _segment_intersect(
            intersect_left_bottom_inner, intersect_bottom_left_inner,
            intersect_left_top_inner, intersect_top_left_inner,
        )
        intersect_left_bottom_inner = intersect
        intersect_left_top_inner = intersect

    v0 = tf.math.subtract(intersect_top_right_inner, intersect_top_left_inner)
    v1 = tf.math.subtract(intersect_right_top_inner, intersect_top_right_inner)
    v2 = tf.math.subtract(intersect_right_bottom_inner, intersect_right_top_inner)
    v3 = tf.math.subtract(intersect_bottom_right_inner, intersect_right_bottom_inner)

    v = tf.concat([v0, v1, v2, v3], axis=-1) / 2

    # We are operating in inverted y axis, since images use inverted y axis.
    # Thus we need to flip the y axis on all results to return correct vectors.
    fix_axis = [[-1., 0.], [0., 1.]]
    v = tf.matmul(fix_axis, v)

    return v


def _segment_intersect(
        seg0_start: tf.Tensor,
        seg0_end: tf.Tensor,
        seg1_start: tf.Tensor,
        seg1_end: tf.Tensor
) -> tf.Tensor:
    """
    Computes the intersection point of two line segments.
    # FIXME: This function assumes that the two segments do intersect.
    Args:
        seg0_start: The starting point of the first segment as a tensor of shape (2, 1).
        seg0_end: The end point of the first segment as a tensor of shape (2, 1).
        seg1_start: The starting point of the second segment as a tensor of shape (2, 1).
        seg1_end: The end point of the second segment as a tensor of shape (2, 1).

    Returns:
        The point of intersection as a tensor of shape (2, 1).
    """
    diff0 = seg0_end - seg0_start
    diff1 = seg1_end - seg1_start

    diff_start = seg1_start - seg0_start

    diff_mat = tf.concat([diff0, diff1], axis=1)
    diff_mat_inv = tf.linalg.inv(diff_mat)
    factors = tf.matmul(diff_mat_inv, diff_start)
    intersection = seg0_start + diff0 * factors[0, 0]
    return intersection
