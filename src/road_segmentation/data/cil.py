import logging
import os
import re
import typing

import numpy as np

import road_segmentation as rs

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
    if len(segmentations.shape) != 4:
        raise ValueError(f'Segmentations must have shape (N, H, W, 1) but are {segmentations.shape}')

    if segmentations.shape[1] % PATCH_SIZE != 0 or segmentations.shape[2] % PATCH_SIZE != 0:
        raise ValueError(f'Width and height must be multiples of {PATCH_SIZE} but got shape {segmentations.shape}')

    labels = np.zeros(
        (segmentations.shape[0], segmentations.shape[1] // PATCH_SIZE, segmentations.shape[2] // PATCH_SIZE, 1)
    )

    # Loop over all patch locations, generating patch labels for all samples at once
    for label_y in range(segmentations.shape[1] // PATCH_SIZE):
        for label_x in range(segmentations.shape[2] // PATCH_SIZE):
            # Calculate input coordinates
            y = label_y * PATCH_SIZE
            x = label_x * PATCH_SIZE

            # Threshold patches to calculate labels
            patches = segmentations[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
            patches_means = np.mean(patches, axis=(1, 2, 3))
            patches_labels = np.where(patches_means > FOREGROUND_THRESHOLD, 1.0, 0.0)

            labels[:, label_y, label_x, 0] = patches_labels

    return labels
