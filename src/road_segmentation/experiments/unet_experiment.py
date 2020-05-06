import argparse
import typing
import numpy as np

import road_segmentation as rs
import tensorflow as tf

import tensorflow_addons as tfa

EXPERIMENT_DESCRIPTION = 'U-Net Baseline'
EXPERIMENT_TAG = 'baseline_unet'

INPUT_PADDING = ((0, 0), (94, 94), (94, 94), (0, 0))
OUTPUT_CROPPING = ((2, 2), (2, 2))


@tf.function
def randomly_shift_image_and_mask(image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly shifts an image and its mask.
    Padding is done via 'REFLECT' strategy, i.e. mirroring

    The shift is chosen randomly.
    Args:
        image: image which should be shifted
        mask: mask which should be shifted

    Returns:
        shifted image and mask
    """
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    height_padding = original_height // 4
    width_padding = original_width // 4
    paddings = [(height_padding, height_padding), (width_padding, width_padding), (0, 0)]
    image = tf.pad(image, paddings, mode="REFLECT")
    mask = tf.pad(mask, paddings, mode="REFLECT")

    height_offset = tf.random.uniform([], 0, image.shape[0] - original_height, dtype=tf.int32)
    width_offset = tf.random.uniform([], 0, image.shape[1] - original_width, dtype=tf.int32)

    image = tf.image.crop_to_bounding_box(image, height_offset, width_offset, original_height, original_width)
    mask = tf.image.crop_to_bounding_box(mask, height_offset, width_offset, original_height, original_width)

    return image, mask


@tf.function
def randomly_rotate_image_and_mask(image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly rotates image and mask.
    Padding is done via 'REFLECT' strategy, i.e. mirroring

    The angle is uniformly drawn.

    Args:
        image: image which should be rotated
        mask: mask which should be rotated by same angle as image

    Returns:
        rotated image and mask
    """
    original_height = tf.shape(image)[0]
    original_width = tf.shape(image)[1]
    paddings = [(original_height - 1, original_height - 1), (original_width - 1, original_width - 1), (0, 0)]
    image = tf.pad(image, paddings, mode="REFLECT")
    mask = tf.pad(mask, paddings, mode="REFLECT")
    random_angles = tf.random.uniform(shape=(), minval=0, maxval=359)
    image = tf.image.resize_with_crop_or_pad(tfa.image.rotate(image, random_angles, interpolation='BILINEAR'),
                                             original_height, original_width)
    mask = tf.image.resize_with_crop_or_pad(tfa.image.rotate(mask, random_angles, interpolation='BILINEAR'),
                                            original_height, original_width)
    return image, mask


@tf.function
def randomly_adjust_image_brightness(image: tf.Tensor,
                                     start_interval: float = -0.2,
                                     end_interval: float = 0.3) -> tf.Tensor:
    """
    Randomly adjust brightness of image by delta from uniform interval between start_interval and end_interval.
    Args:
        image: image which brightness should be adjusted
        start_interval: start of interval to draw delta from
        end_interval: end of interval to draw delta from

    Returns:
        image where brightness is adjusted
    """
    delta = tf.random.uniform([], start_interval, end_interval)
    image = tf.image.adjust_brightness(image, delta)
    return image


@tf.function
def randomly_augment_single_image_and_mask(image: tf.Tensor, mask: tf.Tensor) \
        -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Randomly augments an image and a mask as follows:
        - rotate image and mask by a random angle
        - shift image and mask
        - adjust brightness
    where each transformation is done with probability of 1/2

    Args:
        image: single image
        mask: single mask

    Returns:
        augmented image and mask
    """

    if tf.random.uniform([], minval=0, maxval=1) > 0.5:
        image, mask = randomly_rotate_image_and_mask(image, mask)
    if tf.random.uniform([], minval=0, maxval=1) > 0.5:
        image, mask = randomly_shift_image_and_mask(image, mask)
    if tf.random.uniform([], minval=0, maxval=1) > 0.5:
        image = randomly_adjust_image_brightness(image)

    return image, mask


class BaselineUnetExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
        parser.add_argument('--momentum', type=float, default=0.99, help='Momentum')
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
        parser.add_argument('--apply-dropout', type=bool, default=False,
                            help='Whether or not dropout is applied after conv blocks')
        parser.add_argument('--apply-batch-norm', type=bool, default=False,
                            help='Whether or not batch normalization is applied')
        parser.add_argument('--upsampling-method', type=str, default='transpose',
                            help='"upsampling" for upsampling via interpolation or "transpose" for learnable upsampling'),
        parser.add_argument('--number-of-filters-at-start', type=int, default=64,
                            help='Number of filters at first downsampling block'),
        parser.add_argument('--number-of-scaling-steps', type=int, default=4,
                            help='Number of down and upsampling steps')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'apply_dropout': args.apply_dropout,
            'apply_batch_norm': args.apply_batch_norm,
            'upsampling_method': args.upsampling_method,
            'number_of_filters_at_start': args.number_of_filters_at_start,
            'number_of_scaling_steps': args.number_of_scaling_steps
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']

        self.log.info('Loading training and validation data')
        try:
            trainig_paths, validation_paths = rs.data.cil.train_validation_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(trainig_paths)
            validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            self.log.debug(
                'Loaded %d training and %d validation samples',
                training_images.shape[0],
                validation_images.shape[0]
            )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return
        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.map(randomly_augment_single_image_and_mask)
        training_dataset = training_dataset.batch(batch_size)

        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')

        # FIXME: (if applied to other dataset)
        # INPUT_PADDING and OUTPUT_CROPPING are constants that work exactly for the given training and test data,
        # it will NOT necessarly work for data with a different shape

        model = rs.models.unet.UNet(
            dropout_rate=self.parameters['dropout_rate'],
            apply_dropout=self.parameters['apply_dropout'],
            upsampling_method=self.parameters['upsampling_method'],
            number_of_filters_at_start=self.parameters['number_of_filters_at_start'],
            number_of_scaling_steps=self.parameters['number_of_scaling_steps'],
            apply_batch_norm=self.parameters['apply_batch_norm'],
            input_padding=INPUT_PADDING,
            output_cropping=OUTPUT_CROPPING
        )
        sgd_optimizer = tf.keras.optimizers.SGD(
            momentum=self.parameters['momentum'],
            learning_rate=self.parameters['learning_rate']
        )

        metrics = self.keras.default_metrics(threshold=0.0)

        model.compile(
            optimizer=sgd_optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(),
            self.keras.best_checkpoint_callback(),
            self.keras.log_predictions(validation_images)
        ]

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        tf.keras.utils.plot_model(model, expand_nested=True)
        model.summary()

        self.log.info('Classifier fitted')

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        PREDICTION_THRESHOLD = 0.5

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)

            image = np.asarray([image])
            raw_predictions = classifier.predict(image)
            raw_predictions = tf.sigmoid(raw_predictions)
            prediction_mask = np.where(raw_predictions >= PREDICTION_THRESHOLD, 1, 0)

            prediction_mask_patches = rs.data.cil.segmentation_to_patch_labels(prediction_mask)

            result[sample_id] = prediction_mask_patches[0]

            target_height = image.shape[1] // rs.data.cil.PATCH_SIZE
            target_width = image.shape[2] // rs.data.cil.PATCH_SIZE

            assert result[sample_id].shape == (target_height, target_width)

        return result


def main():
    BaselineUnetExperiment().run()


if __name__ == '__main__':
    main()
