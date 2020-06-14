import argparse
import typing

import numpy as np
import tensorflow as tf
import skimage.color

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FCN Baseline'
EXPERIMENT_TAG = 'baseline_fcn'


def main():
    BaselineFCNExperiment().run()


class BaselineFCNExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are based on Pascal context experiments of original paper
        parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')  # TODO: Should be 16
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'jpu_features': 512,
            'weight_decay': args.weight_decay,
            'output_upsampling': 'nearest',
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'augmentation_interpolation': 'bilinear',
            'training_image_size': (384, 384)
        }

    def fit(self) -> typing.Any:
        # TODO: Data augmentation (implement and test)
        # TODO: Expanding and cropping
        # TODO: Prediction
        # TODO: Learning rate schedule

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
        training_dataset = training_dataset.map(lambda image, mask: self._augment_sample(image, mask))
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        model = TestFastFCN(
            self.parameters['jpu_features'],
            self.parameters['weight_decay'],
            self.parameters['output_upsampling']
        )

        # TODO: This is for testing only
        model.build(training_dataset.element_spec[0].shape)
        model.summary(line_length=120)

        metrics = self.keras.default_metrics(threshold=0.0)

        # TODO: They do weight decay on an optimizer level, not on a case-by-case basis.
        #  There's a difference! tfa has an optimizer-level SGD with weight decay.

        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.parameters['learning_rate'],
                momentum=self.parameters['momentum']
            ),
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

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            # Convert to input colour space
            image = convert_colorspace(image)

            raw_prediction, = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)

            result[sample_id] = prediction

        return result

    def _augment_sample(self, image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        # Gaussian blurring in 50% of the cases
        # TODO

        # Random scaling
        scaling_factor = tf.random.uniform(
            shape=(1,),
            minval=1 - self.parameters['augmentation_max_relative_scaling'],
            maxval=1 + self.parameters['augmentation_max_relative_scaling']
        )[0]
        input_height, input_width, _ = tf.unstack(tf.cast(tf.shape(image), tf.float32))
        scaled_size = tf.cast(
            tf.round((input_height * scaling_factor, input_width * scaling_factor)),
            tf.int32
        )
        scaled_image = tf.image.resize(image, scaled_size, method=self.parameters['augmentation_interpolation'])
        scaled_mask = tf.image.resize(mask, scaled_size, method='nearest')

        # Combine image and mask to ensure same transformations are applied
        concatenated_sample = tf.concat((scaled_image, scaled_mask), axis=-1)

        # Random flip and rotation, this covers all possible permutations which do not require interpolation
        flipped_sample = tf.image.random_flip_left_right(concatenated_sample)
        num_rotations = tf.random.uniform(shape=(1,), minval=0, maxval=3, dtype=tf.int32)[0]
        rotated_sample = tf.image.rot90(flipped_sample, num_rotations)

        # Random crop
        crop_size = self.parameters['training_image_size'] + (4,)  # 3 colour channels + 1 mask channel
        cropped_sample = tf.image.random_crop(rotated_sample, crop_size)

        output_image = cropped_sample[:, :, :3]
        output_mask = cropped_sample[:, :, 3:]

        # Convert mask to labels in {0, 1}
        output_mask = tf.cast(tf.round(output_mask), tf.int32)

        # Convert image to CIE Lab
        # This has to be done after the other transformations since some assume RGB inputs
        [output_image, ] = tf.py_function(convert_colorspace, [output_image], [tf.float32])

        # FIXME: It would make sense to apply colour shifts but the original paper does not

        return output_image, output_mask


def convert_colorspace(images: np.ndarray) -> np.ndarray:
    images_lab = skimage.color.rgb2lab(images)

    # Rescale intensity to [0, 1] and a,b to [-1, 1)
    return images_lab / (100.0, 128.0, 128.0)


class TestFastFCN(tf.keras.models.Model):
    """
    FIXME: This is just a test class
    """

    KERNEL_INITIALIZER = 'he_normal'  # FIXME: This is somewhat arbitrarily chosen

    def __init__(
            self,
            jpu_features: int,
            weight_decay: float,
            output_upsampling: str
    ):
        super(TestFastFCN, self).__init__()

        self.backbone = rs.models.resnet.ResNet50Backbone(weight_decay=weight_decay)
        self.upsampling = rs.models.jpu.JPUModule(
            features=jpu_features,
            weight_decay=weight_decay
        )

        # FIXME: Head is only for testing, replace this with EncNet head
        self.head = rs.models.jpu.FCNHead(
            intermediate_features=256,
            kernel_initializer=self.KERNEL_INITIALIZER,
            weight_decay=weight_decay
        )

        # FIXME: Upsampling of the 8x8 output is slightly unnecessary and should be done more in line with the s16 target
        self.output_upsampling = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=output_upsampling)

        # FIXME: They use an auxiliary FCNHead here to calculate the loss, but never for the output...
        #  Does not really make sense and is also not mentioned in the paper I think
        self.output_crop = tf.keras.layers.Cropping2D(cropping=[[8, 8], [8, 8]])

    def call(self, inputs, training=None, mask=None):
        padded_inputs = tf.pad(
            inputs,
            paddings=[[0, 0], [8, 8], [8, 8], [0, 0]],
            mode='REFLECT'
        )

        intermediate_features = self.backbone(padded_inputs)[-3:]
        upsampled_features = self.upsampling(intermediate_features)
        small_outputs = self.head(upsampled_features)
        padded_outputs = self.output_upsampling(small_outputs)
        outputs = self.output_crop(padded_outputs)
        return outputs


if __name__ == '__main__':
    main()
