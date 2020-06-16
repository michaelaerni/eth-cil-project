import logging
import os
import time
import warnings

import tensorflow as tf
from PIL import Image
from tensorflow_core.python.data.experimental import TFRecordWriter

import road_segmentation as rs

"""
This script process the unsupervised data.
First load .tif files and extract patches, which are then saved as .png files.
Afterwards the .png files get stored in a .tfrecord file per city

The intermediate step of storing the patches as .png is done due to memory constraints.

In the end for each city one .tfrecord file is produced and each contains all patches for that particular city.
The patches are of size 1132x1132 with an overlap of 100 pixel.
"""

_log = logging.getLogger(__name__)

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

    paths_per_city = rs.data.unsupervised.raw_data_paths(
        data_dir)  # get dictionary with path to each .tif image per city
    output_dir = os.path.join(data_dir, 'processed', "unsupervised")

    start = time.time()
    for city in rs.data.unsupervised.CITIES:
        _log.info("Processing {}... (Takes a few minutes)".format(city))
        output_file = os.path.join(output_dir, city + "_new")
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        start_idx = 0
        for i, image_path in enumerate(paths_per_city[city]):
            _log.info("Image {} of {} for {}".format(i, len(paths_per_city[city]), city))
            image = rs.data.cil.load_image(image_path)
            image = image[:, :, :3]  # convert_color_space(np.asarray(image))
            patches = rs.data.unsupervised.extract_patches_from_image(image, overlap, target_height, target_width)
            print(len(patches), patches[-1].shape)
            rs.data.unsupervised.save_images_to_png(patches, output_file, start_idx)
            start_idx += len(patches)

        _log.info("Number of patches for {}: {}".format(city, start_idx))
    _log.info("Process took {} seconds".format(time.time() - start))

    image_paths_per_city = rs.data.unsupervised.preprocessed_png_data_paths_per_city(data_dir)
    _log.info("Found paths for {} cities".format(len(image_paths_per_city)))

    for city in rs.data.unsupervised.CITIES:
        _log.info("Process city: {}...".format(city))
        output_file = os.path.join(data_dir, "processed", "unsupervised",
                                   'new{}.tfrecord'.format(city))
        writer = TFRecordWriter(output_file)
        serialized_features_dataset = tf.data.Dataset.from_generator(
            rs.data.tf_record_util.image_path_generator,
            output_types=tf.string,
            output_shapes=(),
            args=[image_paths_per_city[city]]
        )
        writer.write(serialized_features_dataset)

    _log.info("Finished preprocessing unsupervised data")


def main():
    data_dir = "/media/nic/VolumeAcer/CIL_data"
    target_image_width = 1132
    target_image_height = 1132
    overlap = 100
    preprocess_unsupervised_data(data_dir, target_image_height, target_image_width, overlap)


if __name__ == '__main__':
    main()
