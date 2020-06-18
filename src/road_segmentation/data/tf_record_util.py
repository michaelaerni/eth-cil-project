import time
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_image(image: np.ndarray) -> tf.train.Example:
    """
    Serializes an image
    Args:
        image: the image which needs to be serialized

    Returns:
        image as tf.train.Example
    """
    image = tf.image.encode_png(image)
    feature = {
        'image': _bytes_feature(image)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def images_generator(images):
    """
    Generator which serializes images
    Args:
        images: images in np.array format

    Returns:
        one serialized image
    """
    for i, image in enumerate(images):
        if i % 10 == 0 and i > 0:
            print("Now at {} of {}".format(i, len(images)))
        yield serialize_image(image)


def image_path_generator(image_paths: typing.List[str]) -> tf.train.Example:
    """
    Generator which loads and serializes images given image paths
    Args:
        image_paths: paths to images

    Returns:
        one serialized image
    """
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print("Now at {} of {}".format(i, len(image_paths)))
        image = None
        while image is None:
            try:
                image = rs.data.cil.load_image(image_path)
            except Exception as e:
                print(image_path)
                time.sleep(10)

        image = (image * 255).astype(np.uint8)
        image = serialize_image(image)
        yield image


def parse_image_function(serialized_image: tf.Tensor) -> tf.Tensor:
    """
    Parses an serialized image from a tfrecord
    Args:
        serialized_image: the image which need to be parsed to rgb

    Returns:
        parsed image
    """
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    record = tf.io.parse_single_example(serialized_image, image_feature_description)
    record['image'] = tf.cast(tf.image.decode_image(record['image']), tf.float32) / 255.
    return record['image']
