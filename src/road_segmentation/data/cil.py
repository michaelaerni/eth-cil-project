import logging
import math
import os
import re
import time
import typing
import warnings

import h5py
import matplotlib.image
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow_core.python.data.experimental import TFRecordWriter

import road_segmentation as rs
from road_segmentation import tf_record_util

DATASET_TAG = 'cil-road-segmentation-2020'
PATCH_SIZE = 16
FOREGROUND_THRESHOLD = 0.25
NUM_SAMPLES = 100

_VALIDATION_SPLIT_SEED = 42
_NUM_VALIDATION_SAMPLES = 10

_log = logging.getLogger(__name__)

CITIES = ["Boston", "Dallas", "Detroit", "Houston", "Milwaukee"]


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
    logging.info('Convert color space...')
    converted_images = []
    for image in images:
        # TODO to lab space
        converted_images.append(image[:, :, :3])
    return converted_images


def extract_patches_from_images(images):
    """
    extract patches of one image
    decide on size.
    I guess in the end we want 400x400 pixel,
    so we need 566x566 to be able to rotate them during training data augmentation.
    If we want to do "random shifts" we should extract larger patches

    Where to start:
        maybe start from center and then expand, because border of each (large) image overlaps with other images from same city.
    """
    logging.info('Extract Patches from images...')
    import math
    import matplotlib.pyplot as plt

    def load_img(idx):
        return Image.fromarray(img)

    py = 100
    px = 100
    height = 566 * 2
    width = 566 * 2

    all_patches = []
    for idx in range(len(images)):
        img = load_img(idx)
        Wx = img.width
        flooredx = math.floor((Wx - px) / (width - px))
        # resx = (Wx - flooredx * (width - px) - px) / 2
        Wy = img.height
        flooredy = math.floor((Wy - py) / (height - py))
        # resy = (Wy - flooredy * (height - py) - py) / 2
        # print(resx, flooredx, resy, flooredy)

        maxX = math.floor((Wx - px) / (width - px) - flooredx / 2 + 1)
        maxY = math.floor((Wy - py) / (height - py) - flooredy / 2 + 1)
        minX = int(-(flooredx / 2 - 1))
        minY = int(-(flooredy / 2 - 1))
        # print(maxX, maxY, minX, minY)

        xstart = (Wx - 2 * (width - px) - px) / 2
        ystart = (Wy - 2 * (height - py) - py) / 2
        counter = 0
        # img = load_img(idx)
        # orig_img = load_img(idx)
        for i in range(minY, maxY):
            for j in range(minX, maxX):
                left = xstart + (width - px) * j
                upper = ystart + (height - py) * i
                right = left + width
                lower = upper + height
                counter += 1
                all_patches.append(np.asarray(img.crop((left, upper, right, lower))))
                # draw = ImageDraw.Draw(img)
                # draw.rectangle((left, upper, right, lower), fill=50 + abs(i) * 15 + abs(j) * 25)
                # plt.imshow(img)#.crop((900,700,3200,2400)))
                # plt.show()
                assert np.min((left, upper, right, lower)) >= 0
                assert np.max((left, right)) < img.width
                assert np.max((upper, lower)) < img.height
                # img = load_img()

            # plt.imshow(img)#.crop((900,700,3200,2400)))
            # plt.show()
        plt.imshow(img)  # .crop((900,700,3200,2400)))
        plt.show()
        # img = load_img(idx)
        assert counter == (maxX - minX) * (maxY - minY)
    print(len(all_patches))

    return all_patches


def extract_patches_from_image(image, overlap: int, target_height, target_width):
    """
    extract patches of one image
    decide on size.
    I guess in the end we want 400x400 pixel,
    so we need 566x566 to be able to rotate them during training data augmentation.
    If we want to do "random shifts" we should extract larger patches

    Where to start:
        maybe start from center and then expand, because border of each (large) image overlaps with other images from same city.
    """
    logging.info('Extract Patches from images...')
    py = overlap
    px = overlap

    all_patches = []
    # image = Image.fromarray(image)
    orig_image_width = image.shape[1]
    orig_image_height = image.shape[0]

    flooredx = math.floor((orig_image_width - px) / (target_width - px))
    flooredy = math.floor((orig_image_height - py) / (target_height - py))

    maxX = math.floor((orig_image_width - px) / (target_width - px) - flooredx / 2 + 1)
    maxY = math.floor((orig_image_height - py) / (target_height - py) - flooredy / 2 + 1)
    minX = int(-(flooredx / 2 - 1))
    minY = int(-(flooredy / 2 - 1))

    xstart = (orig_image_width - 2 * (target_width - px) - px) / 2
    ystart = (orig_image_height - 2 * (target_height - py) - py) / 2
    counter = 0
    for i in range(minY, maxY):
        for j in range(minX, maxX):
            left = xstart + (target_width - px) * j
            upper = ystart + (target_height - py) * i
            right = left + target_width
            lower = upper + target_height
            counter += 1

            if np.max((upper, lower)) >= orig_image_height:
                print("why1")
                continue
            if np.max((left, right)) >= orig_image_width:
                print("why2")
                continue
            all_patches.append(image[int(upper):int(lower), int(left):int(right)])
            # exit()
            # all_patches.append(np.asarray(image.crop((left, upper, right, lower))))
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((left, upper, right, lower), fill=50 + abs(i) * 15 + abs(j) * 25)
            # plt.imshow(img)#.crop((900,700,3200,2400)))
            # plt.show()
            assert np.min((left, upper, right, lower)) >= 0
            assert np.max((left, right)) < orig_image_width
            assert np.max((upper, lower)) < orig_image_height

    assert counter == (maxX - minX) * (maxY - minY)

    return all_patches


def preprocess_unsupervised_data(data_dir: str = None,
                                 target_height: int = 1132,
                                 target_width: int = 1132,
                                 overlap: int = 100):
    """
    Main method to run unsupervised data preprocessing
    """
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    paths_per_city = unsupervised_raw_data_paths(
        data_dir)  # get dictionary with path to each .tif image per city
    output_dir = os.path.join(data_dir, 'processed', "unsupervised")

    start = time.time()
    for city in CITIES:
        print("Processing {}... (Takes a few minutes)".format(city))
        output_file = os.path.join(output_dir, city)
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        start_idx = 0
        for i, image_path in enumerate(paths_per_city[city]):
            print("Image {} of {} for {}".format(i, len(paths_per_city[city]), city))
            image = load_image(image_path)
            image = image[:, :, :3]  # convert_color_space(np.asarray(image))
            patches = extract_patches_from_image(image, overlap, target_height, target_width)
            print(len(patches), patches[-1].shape)
            save_images_to_png(patches, output_file, start_idx)
            start_idx += len(patches)
        # output_file = os.path.join(output_dir, f"processed_{city}.tfrecord")
        # save_images_to_tfrecord(patches, output_file)
        logging.info("Number of patches for {}: {}".format(city, start_idx))

    print("Process took {} seconds".format(time.time() - start))


def save_images_to_png(images, output_file, start_idx):
    for i in range(len(images)):
        idx = i + start_idx
        Image.fromarray(images[i]).save(output_file + "/" + str(idx) + ".png")


def save_images_to_tfrecord(images, output_file):
    print(output_file)
    writer = TFRecordWriter(output_file)

    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_record_util.images_generator,
        output_types=tf.string,
        output_shapes=(),
        args=[images]
    )
    writer.write(serialized_features_dataset)


def save_images_to_h5(images, output_file):
    with h5py.File(output_file, 'w') as file:
        _ = file.create_dataset(
            'images', np.shape(images), dtype=h5py.h5t.STD_U8BE, data=images
        )


def load_images_from_h5_files(file_paths):
    """
    I don't know how this method should work.
    Probably depends on how we use h5 in connection with tf dataloader.
    """
    images = []
    for file_path in file_paths:
        file = h5py.File(file_path, "r+")
        images.extend(np.array(file["/images"]))

    return np.asarray(images)


def unsupervised_raw_data_paths(data_dir: str = None):
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
                raise FileNotFoundError(f'Sample satellite image {image_path} not found')

    return paths_per_city


def unsupervised_preprocessed_h5_data_paths(data_directory):
    """
    Return paths to h5 files per city or for all mixed?
    """

    directory = os.path.join(data_directory, "processed", "unsupervised")  # , "processed_Boston.h5")
    a = os.listdir(directory)
    a = [os.path.join(directory, b) for b in a if b.endswith('.h5')]
    return a


def unsupervised_preprocessed_tfrecord_data_paths(data_directory):
    directory = os.path.join(data_directory, "processed", "unsupervised")
    files = os.listdir(directory)
    paths = [os.path.join(directory, file) for file in files if file.endswith('.tfrecord')]
    return paths


def unsupervised_preprocessed_png_data_paths_per_city(data_dir: str = None) -> typing.Dict:
    """
    Returns a dictionary with image paths for each city.
    The key refers to the city and the value is a list of image paths for that city.
    Args:
        data_dir:

    Returns:

    """
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR
    base_directory = os.path.join(data_dir, "processed", "unsupervised")

    image_paths = {}
    for city in CITIES:
        city_directory = os.path.join(base_directory, city)
        files = os.listdir(city_directory)
        image_paths[city] = [os.path.join(city_directory, file) for file in files if file.endswith('.png')]
    return image_paths
