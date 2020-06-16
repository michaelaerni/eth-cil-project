import os
import tensorflow as tf
from tensorflow_core.python.data.experimental import TFRecordWriter

import road_segmentation as rs
from road_segmentation import tf_record_util
from road_segmentation.data.cil import CITIES

data_dir_nic = "/media/nic/VolumeAcer/CIL_data"

data_directory = data_dir_nic
image_paths_per_city = rs.data.cil.unsupervised_preprocessed_png_data_paths_per_city(data_directory)
print(len(image_paths_per_city))

for city in CITIES:
    output_file = os.path.join(data_directory, "processed", "unsupervised",
                               '{}.tfrecord'.format(city))
    writer = TFRecordWriter(output_file)
    print(len(image_paths_per_city[city]))
    serialized_features_dataset = tf.data.Dataset.from_generator(
        tf_record_util.image_path_generator,
        output_types=tf.string,
        output_shapes=(),
        args=[image_paths_per_city[city]]
    )
    writer.write(serialized_features_dataset)
