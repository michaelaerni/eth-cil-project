import argparse
import time
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs
from road_segmentation import tf_record_util

EXPERIMENT_DESCRIPTION = 'Unsupervised Data Pipeline Test'
EXPERIMENT_TAG = 'unsupervised_data_pipeline_test'


def augment_sample(
        image: tf.Tensor,
        max_brightness_delta: float,
        max_shift: int
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Same transformation as in unet_experiment, here just to test the pipeline
    Args:
        image:
        max_brightness_delta:
        max_shift:

    Returns:

    """
    # Randomly shift brightness
    brightness_shifted = tf.image.random_brightness(image, max_delta=max_brightness_delta)

    # Add mask as 4th channel to image to ensure all spatial transformations are equal
    concatenated_sample = tf.concat([brightness_shifted, brightness_shifted], axis=-1)

    # Randomly flip
    flipped = tf.image.random_flip_left_right(concatenated_sample)
    flipped = tf.image.random_flip_up_down(flipped)

    # Randomly crop rotated image to correct shape
    pre_expanded_shape = tf.shape(flipped)
    expanded = tf.pad(
        flipped,
        ((max_shift, max_shift), (max_shift, max_shift), (0, 0)),
        mode='reflect'
    )
    cropped = tf.image.random_crop(expanded, pre_expanded_shape)

    # Separate image and mask again
    output_image = cropped[:, :, :3]
    output_mask = cropped[:, :, 3:]

    return output_image, output_mask


class UnsupervisedDataPipelineExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=5, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
        data_directory = "/media/nic/VolumeAcer/CIL_data"
        self.log.info('Loading training and validation data')
        try:
            image_paths = rs.data.cil.unsupervised_preprocessed_tfrecord_data_paths(data_directory)
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        training_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.interleave(
            tf.data.TFRecordDataset,
            cycle_length=tf.data.experimental.AUTOTUNE,
            block_length=1,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        training_dataset = training_dataset.map(tf_record_util.parse_image_function,
                                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # Just some augmentation
        training_dataset = training_dataset.map(
            lambda image: augment_sample(
                image,
                0.2,
                20
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        training_dataset = training_dataset.batch(batch_size)
        training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        start_time = time.time()
        start_time2 = time.time()
        counter = 0
        for counter, batch in enumerate(training_dataset):
            if counter % 100 == 0:
                print(counter, batch[0].shape, batch[1].shape, time.time() - start_time2)
                start_time2 = time.time()
        print("End", counter, time.time() - start_time)
        print("!!! Exit !!!")
        exit()

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        raise NotImplementedError()


def main():
    UnsupervisedDataPipelineExperiment().run()


if __name__ == '__main__':
    main()
