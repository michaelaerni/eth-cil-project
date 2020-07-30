import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN without context encoding module'
EXPERIMENT_TAG = 'fastfcn_no_context'


def main():
    FastFCNNoContextExperiment().run()


class FastFCNNoContextExperiment(rs.framework.FitExperiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on ADE20k experiments of the original paper
        parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
        parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs')
        parser.add_argument('--prefetch-buffer-size', type=int, default=16, help='Number of batches to pre-fetch')
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
            'weight_decay': 1e-4,
            'head_dropout': 0.1,
            'kernel_initializer': 'he_normal',
            'batch_size': args.batch_size,
            'initial_learning_rate': 1e-2,
            'end_learning_rate': 1e-8,
            'learning_rate_decay': 0.9,
            'momentum': 0.9,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'training_image_size': (384, 384),
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training data')
        try:
            trainig_paths = rs.data.cil.training_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(trainig_paths)
            self.log.debug('Loaded %d samples', training_images.shape[0])
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        training_dataset = training_dataset.map(lambda image, mask: rs.data.cil.augment_image(
            image,
            mask,
            crop_size=self.parameters['training_image_size'],
            max_relative_scaling=self.parameters['augmentation_max_relative_scaling'],
            model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE
        ))
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Build model
        self.log.info('Building model')
        backbone = self._construct_backbone(self.parameters['backbone'])
        model = rs.models.fastfcn.FastFCNNoContext(
            backbone,
            jpu_features=self.parameters['jpu_features'],
            head_dropout_rate=self.parameters['head_dropout'],
            kernel_initializer=self.parameters['kernel_initializer'],
            kernel_regularizer=None
        )

        model.build(training_dataset.element_spec[0].shape)

        # Log model structure if debug logging is enabled
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        metrics = self.keras.default_metrics(threshold=0.0, model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE)

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
            self.keras.best_checkpoint_callback(metric='binary_mean_accuracy')
        ]

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
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

            # Predict labels at model's output stride
            raw_prediction = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1.0, 0.0)

            # Threshold patches to create final prediction
            prediction = rs.data.cil.segmentation_to_patch_labels(prediction, rs.models.fastfcn.OUTPUT_STRIDE)[0]
            prediction = prediction.astype(int)

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
