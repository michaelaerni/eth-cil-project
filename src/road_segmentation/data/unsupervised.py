import itertools
import logging
import os
import re
import typing

import numpy as np
import skimage.util.shape
import tensorflow as tf

import road_segmentation as rs

DATASET_TAG = 'unsupervised'

CITIES = ['Milwaukee', 'Dallas', 'Boston', 'Houston', 'Detroit']
"""
Names of the cities which are available in the unsupervised data
"""

PATCH_HEIGHT = 588
"""
Size of one patch is (PATCH_HEIGHT, PATCH_WIDTH)
"""
PATCH_WIDTH = 588
"""
Size of one patch is (PATCH_HEIGHT, PATCH_WIDTH)
"""

_RAW_INPUT_FILE_REGEX = re.compile(r'^m_.*\.ZIP$')

_log = logging.getLogger(__name__)


def extract_patches_from_image(
        image: np.ndarray,
) -> np.ndarray:
    """
    Extract patches of size (PATCH_HEIGHT, PATCH_WIDTH) from one image.
    Extraction is done, such that as many patches as possible are extracted, where the whole extracted subpart is centered

    Args:
        image: Image from which patches need to be extracted

    Returns:
        NumPy array view with the first axis ranging over patches
         where a single patch has shape (PATCH_HEIGHT, PATCH_WIDTH, 3).
    """
    image_height, image_width, _ = image.shape

    if image_height < PATCH_HEIGHT or image_width < PATCH_WIDTH:
        raise ValueError(f'Image with shape {image.shape} is smaller than patch size')

    # Determine how much has to be cropped from the input image such that it can be exactly divided into tiles
    border_y = image_height % PATCH_HEIGHT
    border_x = image_width % PATCH_WIDTH

    # Center-crop input image
    # If the cropped size is odd then the border on the left/top is one pixel smaller than the right/bottom one
    crop_height = image_height - border_y
    crop_width = image_width - border_x
    offset_y = border_y // 2
    offset_x = border_x // 2
    cropped_image = image[offset_y:(offset_y + crop_height), offset_x:(offset_x + crop_width), :]

    # Return view which ranges over patches
    patch_view = skimage.util.shape.view_as_blocks(cropped_image, block_shape=(PATCH_HEIGHT, PATCH_WIDTH, 3))
    return np.reshape(patch_view, (-1, PATCH_HEIGHT, PATCH_WIDTH, 3))


def raw_data_paths(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns paths for the unsupervised raw data.

    Args:
        data_dir: Base path to data, if none DEFAULT_DATA_DIR is used

    Returns:
        A dictionary, the key refers to the city and the value is a list of image paths for that city.

    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    paths_per_city = {}
    for city in CITIES:
        input_directory = os.path.join(data_dir, 'raw', DATASET_TAG, city)
        if not os.path.isdir(input_directory):
            raise FileNotFoundError(f'Input data directory {input_directory} does not exist')
        _log.info('Appending files from raw input directory %s', input_directory)

        # Filter files to only include files that have the correct name (e.g. ignore summary NAIP_*.ZIP files)
        candidate_files = (
            (file_name, os.path.join(input_directory, file_name))
            for file_name in sorted(os.listdir(input_directory))
        )
        target_paths = [
            target_path
            for file_name, target_path in candidate_files
            if _RAW_INPUT_FILE_REGEX.match(file_name) is not None and os.path.isfile(target_path)
        ]
        _log.debug('Found %d raw input files for city %s', len(target_paths), city)
        paths_per_city[city] = target_paths

    return paths_per_city


def processed_sample_paths(data_dir: str = None) -> typing.List[str]:
    """
    Returns paths to all processed patches of all cities.

    Args:
        data_dir: Base path to data, if none DEFAULT_DATA_DIR is used

    Returns:
        List of paths to all processed patches of the unsupervised data set.
    """

    # Simply concatenate the list of paths over all cities
    return list(itertools.chain(*processed_sample_paths_per_city(data_dir).values()))


def processed_sample_paths_per_city(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns a dictionary with image paths for each city.

    Args:
        data_dir: Base path to data, if none DEFAULT_DATA_DIR is used

    Returns:
        A dictionary, the key refers to the city and the value is a list of image paths for that city.
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_dir = os.path.join(data_dir, 'processed', DATASET_TAG)

    image_paths = {}
    for city in CITIES:
        city_dir = os.path.join(base_dir, city)
        current_paths = []
        for tile_dir_name in os.listdir(city_dir):
            tile_dir = os.path.join(city_dir, tile_dir_name)
            current_paths.extend(
                os.path.join(tile_dir, file)
                for file in os.listdir(tile_dir)
                if file.endswith('.png')
            )
        image_paths[city] = current_paths

    return image_paths


def shuffled_image_dataset(
        paths: typing.List[str],
        output_shape: typing.Optional[typing.Union[tf.TensorShape, typing.List[int], typing.Tuple[int, ...]]] = None,
        seed: typing.Optional[int] = None
) -> tf.data.Dataset:
    """
    Creates a data set which yields the images from the given path in random order.
    Images must be PNGs and are reshuffled each epoch.

    Args:
        paths: Paths of images the resulting data set should contain. Must all be PNG images.
        output_shape: Defines the output shape of the data set elements. If all elements have the same shape then
         setting this value ensures the element_spec of the resulting data set contains the correct shape values.
        seed: Optional seed for shuffling to achieve reproducible results.

    Returns:
        Data set where each entry is a single 3 channel image from the given paths.
    """

    # Perform shuffle before loading images to save memory (i.e. only store all paths instead of all images)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    # FIXME: Make sure the parallelization has the desired effect here
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Ensure output shape if present
    if output_shape is not None:
        def _assert_shape(image: tf.Tensor) -> tf.Tensor:
            image.set_shape(output_shape)
            return image
        dataset = dataset.map(_assert_shape)

    return dataset


def augment_full_sample(
        image: tf.Tensor,
        crop_size: typing.Tuple[int, int, int],
        max_relative_upsampling: float = 0.2,
        interpolation: str = 'bilinear'
) -> tf.Tensor:
    """
    Augment a full unsupervised sample globally, before it is split into query and key.
    Args:
        image: Image to augment (in RGB space).
        crop_size: Output crop size.
        max_relative_upsampling: How much, relative to the input size, the image is randomly upsampled at most.
        interpolation: Interpolation method to be used during rescaling.

    Returns:
        Augmented image (in RGB space).
    """

    # TODO: Compare this with supervised data augmentation, should ideally be quite similar after augmenting patches

    # Random upsampling
    upsampling_factor = tf.random.uniform(
        shape=[],
        minval=1.0,
        maxval=1.0 + max_relative_upsampling
    )
    input_height, input_width, input_channels = tf.unstack(tf.shape(image))
    input_height, input_width = tf.unstack(tf.cast((input_height, input_width), dtype=tf.float32))
    scaled_size = tf.cast(
        tf.round((input_height * upsampling_factor, input_width * upsampling_factor)),
        tf.int32
    )

    upsampled_image = tf.image.resize(image, scaled_size, method=interpolation)

    # Then, random rotate and crop a smaller range from the image
    cropped_sample = rs.data.image.random_rotate_and_crop(
        upsampled_image,
        crop_size[0]
    )

    return cropped_sample


def augment_patch(
        image: tf.Tensor,
        crop_size: typing.Tuple[int, int, int],
        gray_probability: float = 0.1,
        jitter_range: float = 0.2
) -> tf.Tensor:
    """
    Augment a single query or key patch cropped from an unlabelled image.

    Args:
        image: Input RGB patch to augment.
        crop_size: Output crop size.
        gray_probability: Probability with which the image is converted to grayscale.
        jitter_range: Range of jitter applied to hue, saturation, value, and contrast.

    Returns:
        Augmented patch (in CIE Lab) to be used in contrastive learning.
    """

    flipped_sample = tf.image.random_flip_left_right(image)

    cropped_image = rs.data.image.random_rotate_and_crop(
        flipped_sample,
        crop_size[0]
    )

    # Randomly convert to grayscale
    grayscale_sample = rs.data.image.random_grayscale(
        cropped_image,
        probability=gray_probability
    )

    # Random color jitter
    jittered_sample = rs.data.image.random_color_jitter(
        grayscale_sample,
        jitter_range, jitter_range, jitter_range, jitter_range
    )

    # TODO: There is some normalization according to (arXiv:1805.01978 [cs.CV]) happening at the end.
    #  However, those are some random constants whose origin I could not determine yet.
    normalized_sample = jittered_sample

    # Finally, convert to target colorspace
    output_image = rs.data.image.map_colorspace(normalized_sample)

    return output_image


def _load_image(path: tf.Tensor) -> tf.Tensor:
    # Parse to uint8 array
    raw_data = tf.io.read_file(path)
    integer_image = tf.image.decode_png(raw_data)

    # Convert into [0, 1] range
    image = tf.image.convert_image_dtype(integer_image, dtype=tf.float32)

    return image
