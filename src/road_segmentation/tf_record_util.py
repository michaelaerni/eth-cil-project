import numpy as np
import tensorflow as tf

import road_segmentation as rs


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_image(image):
    image = tf.image.encode_png(image)
    feature = {
        'image': _bytes_feature(image)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def images_generator(images):
    for i, image in enumerate(images):
        if i % 10 == 0:
            print("Now at {} of {}".format(i, len(images)))
        if i == 100:
            break
        yield serialize_image(image)


def image_path_generator(image_paths):
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print("Now at {} of {}".format(i, len(image_paths)))
        image = rs.data.cil.load_image(image_path)
        image = (image * 255).astype(np.uint8)
        image = serialize_image(image)
        yield image


def parse_image_function(example_proto):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    record = tf.io.parse_single_example(example_proto, image_feature_description)
    record['image'] = tf.cast(tf.image.decode_image(record['image']), tf.float32) / 255.
    return record['image']


def decode_image_function(record):
    record['image'] = tf.cast(tf.image.decode_image(record['image']), tf.float32) / 255.
    return record['image']
