import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'U-Net Baseline'
EXPERIMENT_TAG = 'baseline_unet'


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
        parser.add_argument('--apply-dropout-after-conv_blocks', type=bool, default=False,
                            help='Whether or not dropout is applied after conv blocks')
        parser.add_argument('--upsampling_method', type=str, default='transpose',
                            help='Either "upsampling" = Upsampling2D or "transpose" = Conv2DTranspose is used in the expansive path')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'apply_dropout_after_conv_blocks': args.apply_dropout_after_conv_blocks,
            'upsampling_method': args.upsampling_method
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
        training_dataset = training_dataset.batch(batch_size)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        model = rs.models.unet.UNet(
            dropout_rate=self.parameters['dropout_rate'],
            apply_dropout_after_conv_blocks=self.parameters['apply_dropout_after_conv_blocks'],
            upsampling_method=self.parameters['upsampling_method']
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

        self.log.info('Classifier fitted')

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)

            target_height = image.shape[0] // rs.data.cil.PATCH_SIZE
            target_width = image.shape[1] // rs.data.cil.PATCH_SIZE

            image = np.asarray([image])
            predictions = classifier.predict(image)
            prediction_mask = tf.sigmoid(predictions)

            prediction_mask_patches = rs.data.cil.segmentation_to_patch_labels(prediction_mask)

            result[sample_id] = prediction_mask_patches[0]
            assert result[sample_id].shape == (target_height, target_width)

        return result


def main():
    BaselineUnetExperiment().run()


if __name__ == '__main__':
    main()
