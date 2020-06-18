import argparse
import logging
import typing

import numpy as np
import skimage.color
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'MoCo Representations Only'
EXPERIMENT_TAG = 'moco_representations'


def main():
    MoCoRepresentationsExperiment().run()


class MoCoRepresentationsExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on the reference implementation at https://github.com/facebookresearch/moco
        # TODO: More parameters if necessary
        parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')  # TODO: Try to increase as much as possible, original is 256
        parser.add_argument('--learning-rate', type=float, default=3e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'backbone': args.backbone,
            'weight_decay': args.weight_decay,
            'kernel_initializer': 'he_normal',  # TODO: Check what the reference implementation uses
            'batch_size': args.batch_size,
            'initial_learning_rate': args.learning_rate,
            'learning_rate_schedule': (120, 160),  # TODO: Check different schedules
            'momentum': args.momentum,
            'epochs': args.epochs,
            'moco_momentum': 0.999,
            'moco_features': 128,
            'moco_temperature': 0.07,
            'moco_queue_size': 65536  # 2^16
            # TODO: Data augmentation parameters
            # 'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            # 'augmentation_interpolation': 'bilinear',
            # 'augmentation_blur_probability': 0.5,
            # 'augmentation_blur_size': 5,  # 5x5 Gaussian filter for blurring
            # TODO: Training image size
            # 'training_image_size': (416, 416)
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training data')
        try:
            training_dataset = self._load_dataset()
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        self.log.info('Building models')
        backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone = self._construct_backbone(self.parameters['backbone'])
        # TODO: Handle MoCo v2 here
        # TODO: Handle initializers and other parameters
        encoder = rs.models.moco.FCHead(
            backbone,
            self.parameters['moco_features'],
            name='encoder'
        )
        momentum_encoder = rs.models.moco.FCHead(
            momentum_backbone,
            self.parameters['moco_features'],
            name='momentum_encoder'
        )
        model = rs.models.moco.EncoderMoCoTrainingModel(
            encoder,
            momentum_encoder,
            self.parameters['moco_momentum'],
            self.parameters['moco_temperature'],
            self.parameters['moco_queue_size'],
            self.parameters['moco_features']
        )

        # Log model structure if debug logging is enabled
        model.build(list(map(lambda spec: spec.shape, training_dataset.element_spec[0])))
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        # Loss is nothing else than the categorical cross entropy with the target class being the true keys
        losses = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = []  # TODO: Metrics

        # TODO: Check whether the implementation is correct
        learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=self.parameters['learning_rate_schedule'],
            values=tuple(
                self.parameters['initial_learning_rate'] * np.power(0.1, idx)
                for idx in range(len(self.parameters['learning_rate_schedule']) + 1)
            )
        )

        # TODO: The paper authors do weight decay on an optimizer level, not on a case-by-case basis.
        #  There's a difference! tfa has an optimizer-level SGD with weight decay.
        #  However, global weight decay might be dangerous if we also have the Encoder head etc.
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate_schedule,
                momentum=self.parameters['momentum']
            ),
            loss=losses,
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(checkpoint_template='{epoch:04d}-{loss:.4f}.h5')
        ] + model.create_callbacks()  # Required MoCo updates

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            callbacks=callbacks
        )

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        # TODO: How to handle prediction in this experiment?
        self.log.warning('Predicting empty masks since no actual segmentation is implemented')
        result = dict()

        for sample_id, image in images.items():
            target_shape = image.shape[0] // rs.data.cil.PATCH_SIZE, image.shape[1] // rs.data.cil.PATCH_SIZE
            result[sample_id] = np.zeros(target_shape)

        return result

    def _load_dataset(self) -> tf.data.Dataset:
        # TODO: Implement
        # TODO: drop_remainder=True in batching is crucial since otherwise, updating the queue fails!
        #  This needs to be kept in mind also for the actual implementation!
        return tf.data.Dataset.from_tensor_slices(
            np.random.uniform(size=(33, 416, 416, 3))  # Use random inputs to test whether the model learns something
        ).map(
            lambda image: ((image, image), 0)  # The target class is always 0, i.e. the positive keys are at index 0
        ).batch(self.parameters['batch_size'], drop_remainder=True)
        raise NotImplementedError()

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        # TODO: ResNet with customizable kernel initializer is not merged yet
        if name == 'ResNet50':
            return rs.models.resnet.ResNet50Backbone(
                weight_decay=self.parameters['weight_decay'],
                #kernel_initializer=self.parameters['kernel_initializer']
            )
        if name == 'ResNet101':
            return rs.models.resnet.ResNet101Backbone(
                weight_decay=self.parameters['weight_decay'],
                #kernel_initializer=self.parameters['kernel_initializer']
            )

        raise AssertionError(f'Unexpected backbone name "{name}"')


def convert_colorspace(images: np.ndarray) -> np.ndarray:
    # TODO: This belongs into the data package
    images_lab = skimage.color.rgb2lab(images)

    # Rescale intensity to [0, 1] and a,b to [-1, 1)
    # FIXME: This might not be the best normalization to do, see the properties of CIE Lab
    return images_lab / (100.0, 128.0, 128.0)


if __name__ == '__main__':
    main()
