import gc
import time

import h5py
import tensorflow as tf
import numpy as np
import argparse

import typing

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Unsupervised Data Pipeline Test'
EXPERIMENT_TAG = 'unsupervised_data_pipeline_test'


def augment_sample(
        image: tf.Tensor,
        max_brightness_delta: float,
        max_shift: int
) -> tf.Tensor:
    # Randomly shift brightness
    brightness_shifted = tf.image.random_brightness(image, max_delta=max_brightness_delta)

    # Randomly flip
    flipped = tf.image.random_flip_left_right(brightness_shifted)
    flipped = tf.image.random_flip_up_down(flipped)

    # Randomly crop rotated image to correct shape
    pre_expanded_shape = tf.shape(flipped)
    expanded = tf.pad(
        flipped,
        ((max_shift, max_shift), (max_shift, max_shift), (0, 0)),
        mode='reflect'
    )
    output_image = tf.image.random_crop(expanded, pre_expanded_shape)

    return output_image


class UnsupervisedDataPipelineExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')

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
            image_paths = rs.data.cil.unsupervised_preprocessed_h5_data_paths(data_directory)
            print(image_paths)
            # images = rs.data.cil.load_images_from_h5_files(image_paths)
            # print(len(images))
            # images = rs.data.cil.load_images(image_paths)
            # validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            # self.log.debug(
            #    'Loaded %d training and %d validation samples',
            #    training_images.shape[0],
            #    validation_images.shape[0]
            # )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            exit()
            return

        """
        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(batch_size)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)
        """
        image_paths = ['/media/nic/VolumeAcer/CIL_data/processed/unsupervised/processed_Boston.h5',
                       '/media/nic/VolumeAcer/CIL_data/processed/unsupervised/processed_Dallas.h5']

        # print(len(rs.data.cil.load_images_from_h5_files([image_paths[0]])))
        # exit()
        file1 = h5py.File(image_paths[0], 'r')
        file2 = h5py.File(image_paths[1], 'r')
        h5_files = {
            image_paths[0]: file1,
            image_paths[1]: file2,
        }
        hf = h5_files[image_paths[0]]
        print(hf["/images"])
        print(len(hf["/images"]))
        print(hf["/images"][0].shape)
        print(hf["/images"][3743].shape)
        #exit()
        def generator(file):
            hf = h5py.File(file.decode(), 'r')
            print("load", file.decode())
            idxs = range(len(hf["/images"]))
            for i in idxs:
                gc.collect()
                image = augment_sample(
                    np.array(hf["/images"][i]),
                    0.2,
                    20
                )
                yield image, image
            hf.close()

        training_dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_paths))

        # You might want to shuffle() the filenames here depending on the application
        training_dataset = training_dataset.interleave(
            lambda filename, _: tf.data.Dataset.from_generator(
                generator,
                (tf.float32, tf.float32),
                output_shapes=(tf.TensorShape([1132, 1132, 3]), tf.TensorShape([1132, 1132, 3])),
                args=(filename,)),
            cycle_length=2,
            block_length=1,
            # num_parallel_calls=4
        )
        # training_dataset = training_dataset.shuffle(buffer_size=256)
        # training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(5)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)
        print(training_dataset.element_spec)

        start_time = time.time()
        start_time2 = time.time()
        counter = 0
        for a in training_dataset:
            counter += 1
            if counter % 100 == 0:
                print(counter, a[0].shape, a[1].shape, time.time() - start_time2)
                start_time2 = time.time()
        print("End", counter, time.time() - start_time)
        exit()

        self.log.info('Building model')
        model = rs.models.test.BaselineCNN(
            dropout_rate=self.parameters['dropout_rate']
        )

        metrics = self.keras.default_metrics(threshold=0.0)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=metrics
        )

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs']
        )

        print("!!! Not fully implemented !!!")
        print("!!! Exit !!!")
        exit()

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        raise NotImplementedError()


def main():
    UnsupervisedDataPipelineExperiment().run()


if __name__ == '__main__':
    main()
