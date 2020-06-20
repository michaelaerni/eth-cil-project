import itertools
import logging
import math
import os
import re
import typing

import numpy as np
import tensorflow as tf
import skimage.util.shape
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
        seed: typing.Optional[int] = None
) -> tf.data.Dataset:
    """
    Creates a data set which yields the images from the given path in random order.
    Images must be PNGs and are reshuffled each epoch.

    Args:
        paths: Paths of images the resulting data set should contain. Must all be PNG images.
        seed: Optional seed for shuffling to achieve reproducible results.

    Returns:
        Data set where each entry is a single 3 channel image from the given paths.
    """

    # Perform shuffle before loading images to save memory (i.e. only store all paths instead of all images)
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    # FIXME: Make sure the parallelization has the desired effect here
    dataset = dataset.map(_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def _load_image(path: tf.Tensor) -> tf.Tensor:
    # Parse to uint8 array
    raw_data = tf.io.read_file(path)
    integer_image = tf.image.decode_png(raw_data)

    # Convert into [0, 1] range
    image = tf.image.convert_image_dtype(integer_image, dtype=tf.float32)

    return image
