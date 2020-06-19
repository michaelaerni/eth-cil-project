import logging
import math
import os
import typing

import numpy as np

import road_segmentation as rs

DATASET_TAG = 'unsupervised'

CITIES = ['Milwaukee', 'Dallas', 'Boston', 'Houston', 'Detroit']

_log = logging.getLogger(__name__)


def extract_patches_from_image(
        image: np.ndarray,
        target_height: int,
        target_width: int
) -> typing.List[np.ndarray]:
    """
    Extract patches of size (target_height, target_width) from one image.
    Extraction is done, such that as many patches as possible are extracted, where the whole extracted subpart is centered
    Args:
        image: Image from which patches need to be extracted
        target_height: Height of one patch
        target_width: Width of one patch

    Returns:
        All patches in a list
    """
    all_patches = []
    orig_image_height = image.shape[0]
    orig_image_width = image.shape[1]

    num_patches_fit_in_width = orig_image_width / target_width
    num_patches_fit_in_height = orig_image_height / target_height

    # Determine start position, such that complete extracted part is centered in original image
    start_position_x = (orig_image_width / 2 - target_width)
    start_position_y = (orig_image_height / 2 - target_height)

    # Start positions are in the middle of the image, so determine how many patches per direction we have to loop over
    max_x = math.ceil(num_patches_fit_in_width / 2)
    max_y = math.ceil(num_patches_fit_in_height / 2)

    for i in range(-max_y, max_y):
        for j in range(-max_x, max_x):
            left = start_position_x + target_width * j
            lower = start_position_y + target_height * i
            right = left + target_width
            upper = lower + target_height

            if np.max((upper, lower)) >= orig_image_height or \
                    np.max((left, right)) >= orig_image_width or \
                    np.min((left, upper, right, lower)) < 0:
                continue

            all_patches.append(image[int(lower):int(upper), int(left):int(right)])

    return all_patches


def raw_data_paths(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns paths for the unsupervised raw .tif data
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    paths_per_city = {}
    for city in CITIES:
        image_dir = os.path.join(data_dir, 'raw', 'unsupervised', city, city)
        _log.debug('Using training sample directory %s', image_dir)

        paths_per_city[city] = [
            (os.path.join(image_dir, file_name))
            for file_name in sorted(os.listdir(image_dir))
            if file_name.endswith('.tif')
        ]
        paths_per_city[city] = sorted(paths_per_city[city])

        # Verify whether all files exist
        for image_path in paths_per_city[city]:
            if not os.path.isfile(image_path) or not os.path.exists(image_path):
                raise FileNotFoundError(f'Image {image_path} not found')

    return paths_per_city


def preprocessed_png_data_paths(data_dir: str = None) -> typing.List[str]:
    """
    Returns paths for all patches in one list.
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_directory = os.path.join(data_dir, 'processed', 'unsupervised')

    image_paths = []
    for city in CITIES:
        city_directory = os.path.join(base_directory, city)
        tile_dirs = os.listdir(city_directory)
        for tile_dir in tile_dirs:
            tile_dir = os.path.join(city_directory, tile_dir)
            files = os.listdir(tile_dir)
            patches_of_tile = [os.path.join(tile_dir, file) for file in files if file.endswith('.png')]
            image_paths.extend(patches_of_tile)
    return image_paths


def preprocessed_png_data_paths_per_city(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns a dictionary with image paths for each city.
    The key refers to the city and the value is a list of image paths for that city.
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_directory = os.path.join(data_dir, 'processed', 'unsupervised')

    image_paths = {}
    for city in CITIES:
        image_paths[city] = []
        city_directory = os.path.join(base_directory, city)
        tile_dirs = os.listdir(city_directory)
        for tile_dir in tile_dirs:
            tile_dir = os.path.join(city_directory, tile_dir)
            files = os.listdir(tile_dir)
            patches_of_tile = [os.path.join(tile_dir, file) for file in files if file.endswith('.png')]
            image_paths[city].extend(patches_of_tile)
    return image_paths
