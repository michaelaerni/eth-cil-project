import logging
import os
import re
import time
import typing
import warnings

import h5py
import matplotlib.image
import numpy as np

import road_segmentation as rs
from PIL import Image

DATASET_TAG = 'cil-road-segmentation-2020'
PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25
NUM_SAMPLES = 100

_VALIDATION_SPLIT_SEED = 42
_NUM_VALIDATION_SAMPLES = 10

_log = logging.getLogger(__name__)


def training_sample_paths(data_dir: str = None) -> typing.List[typing.Tuple[str, str]]:
    """
    Returns a list of tuples of training samples in a fixed order.
    Each tuple contains two file paths, the first one to a satellite image
    and the second one to the corresponding ground truth mask.
    """

    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    sample_dir = os.path.join(data_dir, 'raw', DATASET_TAG, 'training', 'training')
    _log.debug('Using training sample directory %s', sample_dir)

    image_dir = os.path.join(sample_dir, 'images')
    groundtruth_dir = os.path.join(sample_dir, 'groundtruth')

    image_id_regex = re.compile(r'^satImage_\d{3}\.png$')
    result = [
        (os.path.join(image_dir, file_name), os.path.join(groundtruth_dir, file_name))
        for file_name in sorted(os.listdir(image_dir))
        if image_id_regex.match(file_name) is not None
    ]

    # Verify whether all files exist
    for image_path, groundtruth_path in result:
        if not os.path.isfile(image_path) or not os.path.exists(image_path):
            raise FileNotFoundError(f'Sample satellite image {image_path} not found')
        if not os.path.isfile(groundtruth_path) or not os.path.exists(groundtruth_path):
            raise FileNotFoundError(f'Sample groundtruth image {groundtruth_path} not found')

    return result


def train_validation_sample_paths(
        data_dir: str = None
) -> typing.Tuple[typing.List[typing.Tuple[str, str]], typing.List[typing.Tuple[str, str]]]:
    """
    Returns paths for the training and validation samples.
    The split is deterministic and always results in the same validation set.

    Args:
        data_dir: Root data directory. If None then the default one is chosen.

    Returns: Tuple of training sample paths and validation sample paths.

    """

    # Split via IDs
    training_ids, validation_ids = train_validation_split()

    # Load all sample paths
    all_sample_paths = training_sample_paths(data_dir)

    # Match ids to paths
    training_paths = [
        next(filter(lambda entry: entry[0][:-4].endswith(f'{current_id:03d}'), all_sample_paths))
        for current_id in training_ids
    ]
    validation_paths = [
        next(filter(lambda entry: entry[0][:-4].endswith(f'{current_id:03d}'), all_sample_paths))
        for current_id in validation_ids
    ]

    return training_paths, validation_paths


def train_validation_split() -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Split the full training set into training and validation sets.
    The split is deterministic and always results in the same validation set.

    Returns: Tuple of arrays containing training and validations IDs respectively.

    """
    # Encapsulate randomness in a fixed random state which is only used right here
    random_state = np.random.RandomState(_VALIDATION_SPLIT_SEED)

    # Split ids
    all_ids = np.arange(NUM_SAMPLES, dtype=np.int) + 1
    permuted_ids = random_state.permutation(all_ids)
    training_ids = permuted_ids[:-_NUM_VALIDATION_SAMPLES]
    validation_ids = permuted_ids[-_NUM_VALIDATION_SAMPLES:]

    return np.sort(training_ids), np.sort(validation_ids)


def validation_sample_paths(data_dir: str = None) -> typing.List[typing.Tuple[int, str, str]]:
    """
    Returns a sorted list of tuples for validation samples.
    The first entry refers to the id, the second to the satellite image path, the third to the segmentation mask path.
    """

    _, path_tuples = train_validation_sample_paths(data_dir)

    image_id_regex = re.compile(r'.*satImage_(?P<id>\d+)\.png$')
    return [
        (int(image_id_regex.match(image_path).group('id')), image_path, mask_path)
        for image_path, mask_path in path_tuples
    ]


def test_sample_paths(data_dir: str = None) -> typing.List[typing.Tuple[int, str]]:
    """
    Returns a sorted list of tuples for test samples.
    The first entry refers to the id, the second to the satellite image path.
    """

    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    sample_dir = os.path.join(data_dir, 'raw', DATASET_TAG, 'test_images', 'test_images')
    _log.debug('Using test sample directory %s', sample_dir)

    image_id_regex = re.compile(r'^test_(?P<id>\d+)\.png$')
    name_matches = ((file_name, image_id_regex.match(file_name)) for file_name in sorted(os.listdir(sample_dir)))
    return [
        (int(match.group('id')), os.path.join(sample_dir, file_name))
        for file_name, match in name_matches
        if match is not None
    ]


def segmentation_to_patch_labels(segmentations: np.ndarray) -> np.ndarray:
    """
    Converts a binary segmentation mask of a full image into labels over patches
    as is required for the target output.

    Args:
        segmentations: Original segmentations as 4D array with shape (N, H, W, 1).

    Returns:
        Segmentation converted into labels {0, 1} over patches as 4D array.

    """
    # Convert segmentations into patches
    segmentation_patches = cut_patches(segmentations)

    # Threshold mean patch values to generate labels
    patches_means = np.mean(segmentation_patches, axis=(3, 4, 5))
    labels = np.where(patches_means > FOREGROUND_THRESHOLD, 1.0, 0.0)

    return labels


def cut_patches(images: np.ndarray) -> np.ndarray:
    """
    Converts a binary segmentation mask of a full image into labels over patches
    as is required for the target output.

    Args:
        images: Original images as 4D array with shape (N, H, W, C).

    Returns:
        Segmentation converted into patches as 6D array
            with shape (N, H / PATCH_SIZE, W / PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, C).

    """

    # FIXME: This could be implemented more efficiently using some clever NumPy stride tricks

    if len(images.shape) != 4:
        raise ValueError(f'Images must have shape (N, H, W, C) but are {images.shape}')

    if images.shape[1] % PATCH_SIZE != 0 or images.shape[2] % PATCH_SIZE != 0:
        raise ValueError(f'Image width and height must be multiples of {PATCH_SIZE} but got shape {images.shape}')

    num_patches_y = images.shape[1] // PATCH_SIZE
    num_patches_x = images.shape[2] // PATCH_SIZE

    result = np.zeros(
        (images.shape[0], num_patches_y, num_patches_x, PATCH_SIZE, PATCH_SIZE, images.shape[3])
    )

    # Loop over all patch locations, generating patches for all samples at once
    for patch_y in range(num_patches_y):
        for patch_x in range(num_patches_x):
            # Calculate input coordinates
            input_y = patch_y * PATCH_SIZE
            input_x = patch_x * PATCH_SIZE

            # Cut patches
            result[:, patch_y, patch_x, :, :, :] = images[:, input_y:input_y + PATCH_SIZE, input_x:input_x + PATCH_SIZE,
                                                   :]

    return result


def load_images(paths: typing.List[typing.Tuple[str, str]]) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    Load satellite images and segmentation masks from the file system.
    Args:
        paths: List of tuples, each describing a sample. The first entry is the image path, second the mask path.

    Returns:
        Tuple of loaded satellite images and segmentation masks respectively.
            Images are of shape [N, H, W, 3], the masks of shape [N, H, W, 1]
            where all entries are in [0, 1].

    """
    images = []
    masks = []
    for image_path, mask_path in paths:
        images.append(load_image(image_path))
        masks.append(load_image(mask_path))
    images = np.asarray(images)
    masks = np.asarray(masks)

    return images, masks


def load_image(path: str) -> np.ndarray:
    """
    Load a single image from the filesystem.

    Args:
        path: Path to the image file.

    Returns:
        Loaded image as array with shape (H, W, C) where C corresponds to the number of channels (1 or 3).
         Entries are in [0, 1]

    """
    image = matplotlib.image.imread(path)

    if len(image.shape) != 3:
        image = np.expand_dims(image, axis=-1)

    return image


def convert_color_space(images):
    """
    Convert from "RGBx" or whatever format .tif images have, to Lab space
    """
    # RGBs to RGB:
    converted_images = []
    for image in images:
        # TODO to lab space
        converted_images.append(image[:500, :500, :3])
    return converted_images


def extract_patches_from_image(image):
    """
    extract patches of one image
    decide on size.
    I guess in the end we want 400x400 pixel,
    so we need 566x566 to be able to rotate them during training data augmentation.
    If we want to do "random shifts" we should extract larger patches

    Where to start:
        maybe start from center and then expand, because border of each (large) image overlaps with other images from same city.
    """
    # return np.expand_dims(image, 0)
    return image


def preprocess_unsupervised_data(data_dir: str = None):
    """
    Main method to run unsupervised data preprocessing

     Maybe this is helpful:
     https://stackoverflow.com/questions/48309631/tensorflow-tf-data-dataset-reading-large-hdf5-files
     https://www.machinecurve.com/index.php/2020/04/13/how-to-use-h5py-and-keras-to-train-with-data-from-hdf5-files/
    """
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    paths_per_city = unsupervised_raw_data_paths(
        data_dir)  # get dictionary with path to each .tif image per city
    cities = ["Boston", "Dallas", "Detroit", "Houston", "Milwaukee"]
    output_dir = os.path.join(data_dir, 'processed', "unsupervised")

    start = time.time()
    for city in cities:
        print("Processing {}... (Takes a few minutes)".format(city))
        images = []
        for i, image_path in enumerate(paths_per_city[city]):
            images.append(load_image(image_path))
            # for testing read only first five images
            if i == 5:
                break
        images = convert_color_space(np.asarray(images))
        patches = extract_patches_from_image(images)
        patches = np.asarray(patches)
        output_file = os.path.join(output_dir, f"processed_{city}.h5")
        save_images_to_h5(patches, output_file)
        logging.info("Number of patches for {}: {}".format(city, patches.shape[0]))
        # for testing stop after first city
        break
    print("Process took {} seconds".format(time.time() - start))
    exit()
    raise NotImplementedError()


def save_images_to_h5(images, output_file):
    """
    Should store images in h5 format.
    Per city or all together???
    """
    data_type = h5py.special_dtype(vlen=np.dtype('uint8'))
    with h5py.File(output_file, 'w') as file:
        _ = file.create_dataset(
            'images', np.shape(images), dtype=h5py.h5t.STD_U8BE, data=images
        )


def load_images_from_h5(output_file):
    """
    I don't know how this method should work.
    Probably depends on how we use h5 in connection with tf dataloader.
    """
    file = h5py.File(output_file, "r+")
    images = np.array(file["/images"])

    return images


def unsupervised_raw_data_paths(data_dir: str = None):
    """
    Returns paths for the unsupervised raw .tif data
    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    cities = ["Boston", "Dallas", "Detroit", "Houston", "Milwaukee"]

    paths_per_city = {}
    for city in cities:
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
                raise FileNotFoundError(f'Sample satellite image {image_path} not found')

    return paths_per_city


def unsupervised_preprocessed_data_paths(data_directory):
    """
    Return paths to h5 files per city or for all mixed?
    """
    raise NotImplementedError()
