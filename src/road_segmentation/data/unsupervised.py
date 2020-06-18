import logging
import math
import os
import typing

import numpy as np
from PIL import Image

import road_segmentation as rs

DATASET_TAG = 'unsupervised'

CITIES = ["Boston", "Dallas", "Detroit", "Houston", "Milwaukee"]

_log = logging.getLogger(__name__)


def extract_patches_from_image(
        image: np.ndarray,
        target_height: int,
        target_width: int, count) -> typing.List[np.ndarray]:
    """
    Extract patches of size (target_height, target_width) from one one image such that as many patches are extracted as possible.
    """
    #TODO remove count parameter
    _log.info('Extract Patches from images...')
    all_patches = []
    orig_image_width = image.shape[1]
    orig_image_height = image.shape[0]

    num_patches_fit_width = orig_image_width / target_width
    num_patches_fit_height = orig_image_height / target_height

    maxX = math.ceil(num_patches_fit_width - num_patches_fit_width / 2) + 1
    maxY = math.ceil(num_patches_fit_height - num_patches_fit_height / 2) + 1
    minX = int(-(num_patches_fit_width / 2 - 1)) - 1
    minY = int(-(num_patches_fit_height / 2 - 1)) - 1

    xStart = (orig_image_width / 2 - target_width)
    yStart = (orig_image_height / 2 - target_height)
    #TODO remove
    # pil_image = Image.fromarray(image)
    for i in range(minY, maxY):
        for j in range(minX, maxX):
            left = xStart + target_width * j
            upper = yStart + target_height * i
            right = left + target_width
            lower = upper + target_height

            if np.max((upper, lower)) >= orig_image_height or np.max((left, right)) >= orig_image_width:
                continue
            if np.min((left, upper, right, lower)) < 0:
                continue
            #TODO remove
            # draw = ImageDraw.Draw(pil_image)
            # draw.rectangle((left, upper, right, lower), fill=50 + abs(i) * 15 + abs(j) * 25)
            all_patches.append(image[int(upper):int(lower), int(left):int(right)])
            assert np.min((left, upper, right, lower)) >= 0
            assert np.max((left, right)) < orig_image_width
            assert np.max((upper, lower)) < orig_image_height
    # TODO remove
    # plt.imshow(pil_image)
    # plt.savefig("drawing_{}.png".format(count))
    # plt.show()

    return all_patches


def save_images_to_png(images: typing.List[np.ndarray], output_file_path: str, first_idx: int):
    """
    Saves all images as separate .png files
    Uses output_file_path and first_index to generate unique filename
    Args:
        images: list of images which should be stored as .png files
        output_file_path: file path used for each image
        first_idx: index of first file written
    """
    for i in range(len(images)):
        idx = i + first_idx
        Image.fromarray(images[i]).save(output_file_path + "/" + str(idx) + ".png")


def raw_data_paths(data_dir: str = None) -> typing.Dict[str, typing.List[str]]:
    """
    Returns paths for the unsupervised raw .tif data
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    paths_per_city = {}
    for city in CITIES:
        image_dir = os.path.join(data_dir, 'raw', "unsupervised", city, city)
        _log.debug('Using training sample directory %s', image_dir)

        paths_per_city[city] = [
            (os.path.join(image_dir, file_name))
            for file_name in sorted(os.listdir(image_dir))
            if file_name.endswith('.tif')
        ]

        # Verify whether all files exist
        for image_path in paths_per_city[city]:
            if not os.path.isfile(image_path) or not os.path.exists(image_path):
                raise FileNotFoundError(f'Image {image_path} not found')

    return paths_per_city


def preprocessed_h5_data_paths(data_dir: str = None) -> typing.List[str]:
    """
    Return paths to .h5 files
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    directory = os.path.join(data_dir, "processed", "unsupervised")
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files if file.endswith('.h5')]
    return paths


def preprocessed_tfrecord_data_paths(data_dir: str = None) -> typing.List[str]:
    """
    Return paths to all .tfrecord files
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    directory = os.path.join(data_dir, "processed", "unsupervised")
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files if file.endswith('.tfrecord')]
    return paths


def preprocessed_png_data_paths_per_city(data_dir: str = None,
                                         image_height: int = 588,
                                         image_width: int = 588) -> typing.Dict[str, typing.List[str]]:
    """
    Returns a dictionary with image paths for each city.
    The key refers to the city and the value is a list of image paths for that city.
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_directory = os.path.join(data_dir, "processed", "unsupervised")

    image_paths = {}
    for city in CITIES:
        city_directory = os.path.join(base_directory, city + "{}x{}".format(image_height, image_width))
        files = os.listdir(city_directory)
        image_paths[city] = [os.path.join(city_directory, file) for file in files if file.endswith('.png')]
    return image_paths
