import logging
import os
import re
import typing

import matplotlib.image
import numpy as np
import tensorflow as tf

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


def segmentation_to_patch_labels(segmentations: np.ndarray, model_output_stride: int = 1) -> np.ndarray:
    """
    Converts a binary segmentation mask of a full image into labels over patches
    as is required for the target output.

    Args:
        segmentations: Original segmentations as 4D array with shape (N, H, W, 1).
        model_output_stride: The output stride of the model that produced the segmentations.

    Returns:
        Segmentation converted into labels {0, 1} over patches as 4D array.

    """
    # Convert segmentations into patches
    segmentation_patches = cut_patches(segmentations, model_output_stride=model_output_stride)

    # Threshold mean patch values to generate labels
    patches_means = np.mean(segmentation_patches, axis=(3, 4, 5))
    labels = np.where(patches_means > FOREGROUND_THRESHOLD, 1.0, 0.0)

    return labels


def cut_patches(images: np.ndarray, model_output_stride: int = 1) -> np.ndarray:
    """
    Converts a binary segmentation mask of a full image into labels over patches
    as is required for the target output.

    Args:
        images: Original images as 4D array with shape (N, H, W, C).
        model_output_stride: The output stride of the model that produced the segmentations.

    Returns:
        Segmentation converted into patches as 6D array
            with shape (N, H / (PATCH_SIZE//model_output_stride), W / (PATCH_SIZE//model_output_stride), (PATCH_SIZE//model_output_stride), (PATCH_SIZE//model_output_stride), C).
    """

    model_cut_patch_size = PATCH_SIZE//model_output_stride

    # FIXME: This could be implemented more efficiently using some clever NumPy stride tricks

    if len(images.shape) != 4:
        raise ValueError(f'Images must have shape (N, H, W, C) but are {images.shape}')

    if images.shape[1] % model_cut_patch_size != 0 or images.shape[2] % model_cut_patch_size != 0:
        raise ValueError(f'Image width and height must be multiples of {model_cut_patch_size} but got shape {images.shape}')

    num_patches_y = images.shape[1] // model_cut_patch_size
    num_patches_x = images.shape[2] // model_cut_patch_size

    result = np.zeros(
        (images.shape[0], num_patches_y, num_patches_x, model_cut_patch_size, model_cut_patch_size, images.shape[3])
    )

    # Loop over all patch locations, generating patches for all samples at once
    for patch_y in range(num_patches_y):
        for patch_x in range(num_patches_x):
            # Calculate input coordinates
            input_y = patch_y * model_cut_patch_size
            input_x = patch_x * model_cut_patch_size

            # Cut patches
            result[:, patch_y, patch_x, :, :, :] = images[:, input_y:input_y + model_cut_patch_size, input_x:input_x + model_cut_patch_size, :]

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


def augment_image(
        image: tf.Tensor,
        mask: tf.Tensor,
        crop_size: typing.Tuple[int, int, int],  # TODO: Might vary between models
        max_relative_scaling: float,  # TODO: Might vary between models
        blur_probability: float = 0.5,
        blur_kernel_size: int = 5,
        interpolation: str = 'bilinear',
        model_output_stride: int = 1
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Augments a single sample (image segmentation pair).

    Args:
        image: Image to augment in RGB color space.
        mask: Corresponding segmentation mask.
        max_relative_scaling: How much, relative to the input size, the image is scaled up or down randomly.
        crop_size: Output crop size.
        blur_probability: Probability with which a Gaussian blur is applied to the image.
        blur_kernel_size: Size of the blur kernel.
        interpolation: Interpolation used to resample the image after scaling.
        model_output_stride: Segmentation will be downsampled to this stride after applying augmentations.

    Returns:
        Augmented sample in CIE Lab space.
    """
    # Random Gaussian blurring
    do_blur = tf.random.uniform(shape=[], dtype=tf.float32) < blur_probability
    blurred_image = tf.cond(do_blur, lambda: rs.data.image.random_gaussian_blur(image, blur_kernel_size), lambda: image)
    blurred_image.set_shape(image.shape)  # Must set shape manually since it cannot be inferred from tf.cond

    # Random scaling
    scaling_factor = tf.random.uniform(
        shape=[],
        minval=1.0 - max_relative_scaling,
        maxval=1.0 + max_relative_scaling
    )
    input_height, input_width, _ = tf.unstack(tf.cast(tf.shape(blurred_image), tf.float32))
    scaled_size = tf.cast(
        tf.round((input_height * scaling_factor, input_width * scaling_factor)),
        tf.int32
    )
    scaled_image = tf.image.resize(blurred_image, scaled_size, method=interpolation)
    scaled_mask = tf.image.resize(mask, scaled_size, method='nearest')  # TODO: This will have to change if correct output strides are implemented

    # Combine image and mask to ensure same transformations are applied
    concatenated_sample = tf.concat((scaled_image, scaled_mask), axis=-1)

    # Random flip and rotation, this covers all possible permutations which do not require interpolation
    flipped_sample = tf.image.random_flip_left_right(concatenated_sample)
    num_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    rotated_sample = tf.image.rot90(flipped_sample, num_rotations)

    # Random crop
    actual_crop_size = crop_size + (4,)  # 3 colour channels + 1 mask channel
    cropped_sample = tf.image.random_crop(rotated_sample, actual_crop_size)

    # Split combined image and mask again
    cropped_image = cropped_sample[:, :, :3]
    output_mask_large = cropped_sample[:, :, 3:]

    # Convert mask to labels in {0, 1} but keep as floats
    output_mask = resize_mask_to_stride(output_mask_large, model_output_stride)
    output_mask = tf.round(output_mask)

    # Convert image to CIE Lab
    # This has to be done after the other transformations since some assume RGB inputs
    output_image_lab = rs.data.image.map_colorspace(cropped_image)

    return output_image_lab, output_mask


def resize_mask_to_stride(mask: tf.Tensor, stride: int) -> tf.Tensor:
    """
    Resizes a single mask down to a given stride using nearest neighbor.
    Args:
        mask: The input mask.
        stride: The stride to which the input mask is to be down sampled.

    Returns:
        The downsampled mask.
    """

    expanded = False
    if len(tf.shape(mask)) == 3:
        expanded = True
        mask = tf.expand_dims(mask, axis=0)

    _, width, height, _ = tf.unstack(tf.shape(mask))
    output_mask = tf.image.resize(
        mask,
        (width//stride, height//stride),
        method='nearest'
    )
    if expanded:
        output_mask = tf.squeeze(output_mask, axis=0)
    return output_mask

