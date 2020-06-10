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

        training_masks = rs.data.cil.segmentation_to_patch_labels(training_masks)
        validation_masks = rs.data.cil.segmentation_to_patch_labels(validation_masks)

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        inputs = tf.keras.layers.Input(shape=training_dataset.element_spec[0].shape[1:])
        # TODO: Padding is only for testing purposes and should not be done in the real model
        padded_inputs = tf.pad(
            inputs,
            paddings=[[0, 0], [8, 8], [8, 8], [0, 0]],
            mode='REFLECT'
        )
        backbone = rs.models.resnet.ResNet50Backbone()
        upsampling = rs.models.jpu.JPUModule(
            features=self.parameters['jpu_features'],
            weight_decay=self.parameters['jpu_weight_decay']
        )
        # TODO: Head is only for testing
        head = tf.keras.layers.Conv2D(filters=1, kernel_size=1, activation=None)
        # TODO: Upsampling of the 8x8 output is slightly unnecessary and should be done more in line with the s16 target
        output_upsampling = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=self.parameters['output_upsampling'])
        # TODO: Something seems broken
        outputs = output_upsampling(head(upsampling(backbone(padded_inputs)[-3:])))
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
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


def main():
    BaselineFCNExperiment().run()


if __name__ == '__main__':
    main()
