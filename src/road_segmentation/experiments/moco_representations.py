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
        parser.add_argument('--batch-size', type=int, default=96, help='Training batch size')  # TODO: Try to increase as much as possible, original is 256
        parser.add_argument('--learning-rate', type=float, default=3e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        parser.add_argument('--prefetch-buffer-size', type=int, default=16, help='Number of batches to pre-fetch')  # FIXME: What would be a sensible default?
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
            'nesterov': True,
            'initial_learning_rate': args.learning_rate,
            'learning_rate_schedule': (120, 160),  # TODO: Check different schedules
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_features': 128,
            'moco_temperature': 0.07,
            'moco_queue_size': 16384 - 64,  # TODO: Originally was 65536 (2^16)
            # TODO: Data augmentation parameters
            # 'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            # 'augmentation_interpolation': 'bilinear',
            # 'augmentation_blur_probability': 0.5,
            # 'augmentation_blur_size': 5,  # 5x5 Gaussian filter for blurring
            'training_image_size': (224, 224, 3)  # TODO: Decide on an image size
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training data')
        try:
            training_dataset = self._load_dataset()
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Make sure the batch size and queue size are compatibale
        assert self.parameters['moco_queue_size'] % self.parameters['batch_size'] == 0,\
            'Queue size must be a multiple of the batch size'

        self.log.info('Building models')
        backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone.trainable = False
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
            name='momentum_encoder',
            trainable=False
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
        # TODO: Metrics (other and/or more sensible ones)
        # TODO: More evaluation during training
        metrics = [
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_5_categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_top_1_categorical_accuracy')
        ]

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
        # TODO: Could use target_tensors here to specify the (constant) targets instead of feeding them via dataset
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate_schedule,
                momentum=self.parameters['momentum'],
                nesterov=self.parameters['nesterov']
            ),
            loss=losses,
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(checkpoint_template='{epoch:04d}-{loss:.4f}.h5')
        ] + model.create_callbacks()  # Required MoCo updates

        # Fit model
        # TODO: GPU utilization seems to be quite low still, investigate why that is the case!
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
        # Uses all unsupervised samples in a flat order
        dataset = rs.data.unsupervised.shuffled_image_dataset(
            rs.data.unsupervised.processed_sample_paths(self.parameters['base_data_directory']),
            seed=self.SEED
        )

        # TODO: Implement data augmentation in this step here
        dataset = dataset.map(lambda image: (
            tf.image.random_crop(image, self.parameters['training_image_size']),
            tf.image.random_crop(image, self.parameters['training_image_size'])
        ))

        # Add label and convert images to correct colour space
        # The target class is always 0, i.e. the positive keys are at index 0
        dataset = dataset.map(
            lambda image1, image2: ((convert_colorspace(image1), convert_colorspace(image2)), 0)
        )

        # Batch samples
        # drop_remainder=True is crucial since the sample queue assumes queue size modulo batch size to be 0
        dataset = dataset.batch(self.parameters['batch_size'], drop_remainder=True)

        # Prefetch batches to decrease latency
        dataset = dataset.prefetch(self.parameters['prefetch_buffer_size'])

        return dataset

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


@tf.function
def convert_colorspace(images: tf.Tensor) -> tf.Tensor:
    # TODO: This belongs into the data package
    [images_lab, ] = tf.py_function(skimage.color.rgb2lab, [images], [tf.float32])

    # Make sure shape information is correct after py_function call
    images_lab.set_shape(images.get_shape())

    # Rescale intensity to [0, 1] and a,b to [-1, 1)
    # FIXME: This might not be the best normalization to do, see the properties of CIE Lab
    return images_lab / (100.0, 128.0, 128.0)


if __name__ == '__main__':
    main()
