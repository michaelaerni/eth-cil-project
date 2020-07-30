import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'U-Net Baseline'
EXPERIMENT_TAG = 'baseline_unet'

INPUT_PADDING = ((94, 94), (94, 94))


class BaselineUNetExperiment(rs.framework.FitExperiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
        parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--momentum', type=float, default=0.99, help='Momentum')
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
        parser.add_argument(
            '--dropout-rate', type=float, default=0.5,
            help='Dropout rate for features at the end of the contracting path and bottleneck'
        )
        parser.add_argument(
            '--apply-batch-norm', action='store_true', help='If specified then batch normalization is applied'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            # Note that this is equivalent to Xavier which the paper specifies.
            # However, this converges very very slowly, using e.g. he_normal is much faster.
            'kernel_initializer': 'glorot_normal',
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'dropout_rate': args.dropout_rate,
            'apply_batch_norm': args.apply_batch_norm,
            'prefetch_buffer_size': 16,
            'augmentation_max_brightness_delta': 0.2,
            'augmentation_max_shift': 20
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']

        self.log.info('Loading training data')
        try:
            trainig_paths = rs.data.cil.training_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(trainig_paths)
            self.log.debug('Loaded %d training samples', training_images.shape[0])
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            raise

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        # Use U-Net specific data augmentation
        training_dataset = training_dataset.map(
            lambda image, mask: augment_sample(
                image,
                mask,
                self.parameters['augmentation_max_brightness_delta'],
                self.parameters['augmentation_max_shift']
            )
        )
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Build model
        self.log.info('Building model')
        model = rs.models.unet.UNet(
            input_padding=INPUT_PADDING,
            apply_batch_norm=self.parameters['apply_batch_norm'],
            dropout_rate=self.parameters['dropout_rate'],
            kernel_initializer=self.parameters['kernel_initializer']
        )
        optimizer = tf.keras.optimizers.SGD(
            momentum=self.parameters['momentum'],
            learning_rate=self.parameters['learning_rate']
        )

        metrics = self.keras.default_metrics(threshold=0.0)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(checkpoint_template='{epoch:04d}-{loss:.4f}.h5'),
            self.keras.best_checkpoint_callback(metric='binary_mean_accuracy')
        ]

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            callbacks=callbacks
        )

        self.log.info('Classifier fitted')

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            # Predict labels at model's output stride
            raw_prediction = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1.0, 0.0)

            # Threshold patches to create final prediction
            prediction = rs.data.cil.segmentation_to_patch_labels(prediction)[0]
            prediction = prediction.astype(int)

            result[sample_id] = prediction

        return result


def augment_sample(
        image: tf.Tensor,
        mask: tf.Tensor,
        max_brightness_delta: float,
        max_shift: int
) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly augments a single input sample (image and mask combination) via random brightness shift,
    random flips and random shift followed by a crop.

    Args:
        image: 3D RGB image to augment with shape (H, W, 3).
        mask: Corresponding mask to augment with shape (H, W, 1).
        max_brightness_delta: Maximum brightness offset in [0, 1].
        max_shift: Maximum shift of image (with reflected padding).

    Returns:
        Augmented sample as a tuple (image, mask) with same shapes and semantics as inputs.
    """

    # Randomly shift brightness
    brightness_shifted = tf.image.random_brightness(image, max_delta=max_brightness_delta)

    # Add mask as 4th channel to image to ensure all spatial transformations are equal
    concatenated_sample = tf.concat([brightness_shifted, mask], axis=-1)

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


def main():
    BaselineUNetExperiment().run()


if __name__ == '__main__':
    main()
