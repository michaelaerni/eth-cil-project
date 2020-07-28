import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN Baseline without context encoding module'
EXPERIMENT_TAG = 'baseline_fastfcn_no_context'


def main():
    BaselineFCNNoContextExperiment().run()


class BaselineFCNNoContextExperiment(rs.framework.FitExperiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on ADE20k experiments of the original paper
        parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')  # FIXME: Was 16 originally
        parser.add_argument('--learning-rate', type=float, default=1e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        # TODO: Adjust after search
        return {
            'jpu_features': 512,
            'backbone': args.backbone,
            'weight_decay': args.weight_decay,
            'head_dropout': 0.1,
            'output_upsampling': 'nearest',
            'kernel_initializer': 'he_normal',  # FIXME: This might not necessarily be the best choice
            'batch_size': args.batch_size,
            'initial_learning_rate': args.learning_rate,
            'end_learning_rate': 1e-8,  # FIXME: The original authors decay to zero but small non-zero might be better
            'learning_rate_decay': 0.9,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': 16,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'training_image_size': (384, 384),
        }

    def fit(self) -> typing.Any:
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
        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        training_dataset = training_dataset.map(lambda image, mask: rs.data.cil.augment_image(
            image,
            mask,
            crop_size=self.parameters['training_image_size'],
            max_relative_scaling=self.parameters['augmentation_max_relative_scaling']
        ))
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Validation images can be directly converted to the model colour space
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(validation_images), validation_masks)
        )
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        backbone = self._construct_backbone(self.parameters['backbone'])
        model = rs.models.fastfcn.FastFCNNoContext(
            backbone,
            self.parameters['jpu_features'],
            self.parameters['head_dropout'],
            self.parameters['kernel_initializer'],
            self.parameters['output_upsampling'],
            kernel_regularizer=None
        )

        model.build(training_dataset.element_spec[0].shape)

        # Log model structure if debug logging is enabled
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        metrics = self.keras.default_metrics(threshold=0.0)

        # TODO: Check whether the binary cross-entropy loss behaves correctly
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        steps_per_epoch = np.ceil(training_images.shape[0] / self.parameters['batch_size'])
        self.log.debug('Calculated steps per epoch: %d', steps_per_epoch)

        optimizer = self.keras.build_optimizer(
            total_steps=self.parameters['epochs'] * steps_per_epoch,
            initial_learning_rate=self.parameters['initial_learning_rate'],
            end_learning_rate=self.parameters['end_learning_rate'],
            learning_rate_decay=self.parameters['learning_rate_decay'],
            momentum=self.parameters['momentum'],
            weight_decay=self.parameters['weight_decay']
        )
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(),
            self.keras.best_checkpoint_callback(),
            self.keras.log_predictions(
                validation_images=rs.data.image.rgb_to_cielab(validation_images),
                display_images=validation_images
            )
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

            # Convert to model colour space
            image = rs.data.image.rgb_to_cielab(image)

            raw_prediction = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)

            result[sample_id] = prediction

        return result

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        if name == 'ResNet50':
            return rs.models.resnet.ResNet50Backbone(
                kernel_initializer=self.parameters['kernel_initializer']
            )
        if name == 'ResNet101':
            return rs.models.resnet.ResNet101Backbone(
                kernel_initializer=self.parameters['kernel_initializer']
            )

        raise AssertionError(f'Unexpected backbone name "{name}"')


if __name__ == '__main__':
    main()
