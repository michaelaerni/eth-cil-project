import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FCN Baseline'
EXPERIMENT_TAG = 'baseline_fcn'


class BaselineFCNExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')  # TODO: Adjust
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')  # TODO: Adjust
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')  # TODO: Adjust

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'jpu_features': 512,
            'jpu_weight_decay': 1e-4,
            'output_upsampling': 'nearest',
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        # TODO: Data augmentation

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
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        model = TestFastFCN(
            self.parameters['jpu_features'],
            self.parameters['jpu_weight_decay'],
            self.parameters['output_upsampling']
        )
        model.build(training_dataset.element_spec[0].shape)
        model.summary(line_length=200)

        metrics = self.keras.default_metrics(threshold=0.0)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
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

            raw_prediction, = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)

            result[sample_id] = prediction

        return result


class TestFastFCN(tf.keras.models.Model):
    """
    FIXME: This is just a test class
    """

    KERNEL_INITIALIZER = 'he_normal'  # FIXME: This is somewhat arbitrarily chosen

    def __init__(
            self,
            jpu_features: int,
            jpu_weight_decay: float,
            output_upsampling: str
    ):
        super(TestFastFCN, self).__init__()

        self.backbone = rs.models.resnet.ResNet50Backbone()
        self.upsampling = rs.models.jpu.JPUModule(
            features=jpu_features,
            weight_decay=jpu_weight_decay
        )

        # FIXME: Head is only for testing, replace this with EncNet head
        self.head = rs.models.jpu.FCNHead(intermediate_features=256, kernel_initializer=self.KERNEL_INITIALIZER)

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


def main():
    BaselineFCNExperiment().run()


if __name__ == '__main__':
    main()
