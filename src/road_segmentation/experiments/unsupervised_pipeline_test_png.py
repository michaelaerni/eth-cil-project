import argparse
import time
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Unsupervised Data Pipeline Test png'
EXPERIMENT_TAG = 'unsupervised_data_pipeline_test_png'


def augment_sample(
        image: tf.Tensor,
        max_brightness_delta: float,
        max_shift: int
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Same transformation as in unet_experiment, here just to test the pipeline
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


def decode_img(image: tf.Tensor) -> tf.Tensor:
    """
    Decodes an RGB image from a string
    Args:
        image: image as string

    Returns:
        decode image
    """
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def process_path(file_path: tf.Tensor):
    """
    Load the raw data from the file as a string
    Args:
        file_path: path to file

    Returns:
        image as tensor
    """
    image = tf.io.read_file(file_path)
    image = decode_img(image)
    return image


class UnsupervisedPNGDataPipelineExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=5, help='Training batch size')
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
        data_directory = "/media/nic/VolumeAcer/CIL_data"
        self.log.info('Loading training and validation data')
        try:
            image_paths = rs.data.unsupervised.preprocessed_png_data_paths(data_directory)
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        print(len(image_paths))

        training_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        training_dataset = training_dataset.shuffle(buffer_size=len(image_paths))
        training_dataset = training_dataset.map(process_path,
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
        # FIXME: maybe prefetch helps to speed data loading up, need to be tested
        # training_dataset = training_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        start_time = time.time()
        batch_time = time.time()
        counter = 0
        for counter, batch in enumerate(training_dataset):
            if counter % 100 == 0 and counter != 0:
                print(counter, batch[0].shape, batch[1].shape, time.time() - batch_time)
                batch_time = time.time()
        print("End", counter, time.time() - start_time)
        print("!!! Exit !!!")
        exit()

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        raise NotImplementedError()


def main():
    UnsupervisedPNGDataPipelineExperiment().run()


if __name__ == '__main__':
    main()
