import logging
import math
import os
import re
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

DATASET_TAG = 'unsupervised'

CITIES = ['Milwaukee', 'Dallas', 'Boston', 'Houston', 'Detroit']
"""
Names of the cities which are available in the unsupervised data
"""

PATCH_HEIGHT = 588
"""
Size of on patch is (PATCH_HEIGHT, PATCH_WIDTH)
"""
PATCH_WIDTH = 588
"""
Size of on patch is (PATCH_HEIGHT, PATCH_WIDTH)
"""

_RAW_INPUT_FILE_REGEX = re.compile(r'^m_.*\.ZIP$')

_log = logging.getLogger(__name__)


def extract_patches_from_image(
        image: np.ndarray,
) -> typing.List[np.ndarray]:
    """
    Extract patches of size (PATCH_HEIGHT, PATCH_WIDTH) from one image.
    Extraction is done, such that as many patches as possible are extracted, where the whole extracted subpart is centered

    Args:
        image: Image from which patches need to be extracted

    Returns:
        All patches in a list
    """
    all_patches = []
    orig_image_height = image.shape[0]
    orig_image_width = image.shape[1]

    num_patches_fit_in_width = orig_image_width / PATCH_WIDTH
    num_patches_fit_in_height = orig_image_height / PATCH_HEIGHT

    # Determine start position, such that complete extracted part is centered in original image
    start_position_x = (orig_image_width / 2 - PATCH_WIDTH)
    start_position_y = (orig_image_height / 2 - PATCH_HEIGHT)

    # Start positions are in the middle of the image, so determine how many patches per direction we have to loop over
    max_x = math.ceil(num_patches_fit_in_width / 2)
    max_y = math.ceil(num_patches_fit_in_height / 2)

    for i in range(-max_y, max_y):
        for j in range(-max_x, max_x):
            left = start_position_x + PATCH_WIDTH * j
            lower = start_position_y + PATCH_HEIGHT * i
            right = left + PATCH_WIDTH
            upper = lower + PATCH_HEIGHT

            if np.max((upper, lower)) >= orig_image_height or \
                    np.max((left, right)) >= orig_image_width or \
                    np.min((left, upper, right, lower)) < 0:
                continue

            all_patches.append(image[int(lower):int(upper), int(left):int(right)])

    return all_patches


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
    Returns paths for all patches in one list.

    Args:
        data_dir: Base path to data, if none DEFAULT_DATA_DIR is used

    Returns:
        A list of sample paths
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    image_paths_per_city = preprocessed_data_paths_per_city(data_dir)
    image_paths = []
    for city in image_paths_per_city:
        image_paths.extend(image_paths_per_city[city])

    return image_paths


def preprocessed_data_paths_per_city(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns a dictionary with image paths for each city.

    Args:
        data_dir: Base path to data, if none DEFAULT_DATA_DIR is used

    Returns:
        A dictionary, the key refers to the city and the value is a list of image paths for that city.
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_directory = os.path.join(data_dir, 'processed', DATASET_TAG)

    image_paths = {}
    for city in CITIES:
        city_directory = os.path.join(base_directory, city)
        patches_of_city = []
        for tile_dir in os.listdir(city_directory):
            tile_dir = os.path.join(city_directory, tile_dir)
            patches_of_tile = [os.path.join(tile_dir, file) for file in os.listdir(tile_dir) if file.endswith('.png')]
            patches_of_city.extend(patches_of_tile)
        image_paths[city] = patches_of_city
    return image_paths


def decode_img(image: tf.Tensor) -> tf.Tensor:
    """
    Decodes an RGB image from a string

    Args:
        image: Image as string
    Returns:
        Decoded image
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def load_image(file_path: tf.Tensor):
    """
    Load the raw data from the file as a string

    Args:
        file_path: Path to file
    Returns:
        Decoded image as tensor
    """
    image = tf.io.read_file(file_path)
    image = decode_img(image)
    return image
