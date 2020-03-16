import logging
import os
import re
import typing

import road_segmentation as rs

DATASET_TAG = 'cil-road-segmentation-2020'


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
