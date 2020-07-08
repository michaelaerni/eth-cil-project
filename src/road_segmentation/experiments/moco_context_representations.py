import argparse
import logging
import typing
import os

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'MoCo Representations Only'
EXPERIMENT_TAG = 'moco_context_representations'


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
        parser.add_argument('--batch-size', type=int, default=16,
                            help='Training batch size')  # FIXME: This is the max fitting on a 1080Ti, original is 256
        parser.add_argument('--learning-rate', type=float, default=3e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        parser.add_argument('--prefetch-buffer-size', type=int, default=16,
                            help='Number of batches to pre-fetch')  # FIXME: What would be a sensible default?
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )
        parser.add_argument(
            '--moco-head',
            type=str,
            default='fc',
            choices=('mlp', 'fc'),
            help='MoCo head to use, mlp is MoCo v2 where fc is MoCo v1'
        )

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'backbone': args.backbone,
            'weight_decay': args.weight_decay,
            # FIXME: Keras uses glorot_uniform for both initializers.
            #  he_uniform is the same as the PyTorch default with an additional (justified) factor sqrt(6).
            #  Generally, there is no principled way to decide uniform vs normal.
            #  Also, He "should work better for ReLU" compared to Glorot but that is also not very clear.
            #  We should decide on which one to use.
            'kernel_initializer': 'he_uniform',
            'dense_initializer': 'he_uniform',
            'output_upsampling': 'nearest',
            'jpu_features': 512,
            'batch_size': args.batch_size,
            'nesterov': True,
            'initial_learning_rate': args.learning_rate,
            'learning_rate_schedule': (120, 160),  # TODO: Test different schedules
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_features': 128,
            'moco_temperature': 0.07,
            'moco_queue_size': 16384,  # TODO: Originally was 65536 (2^16)
            'moco_head': args.moco_head,
            'moco_mlp_features': 2048,
            # FIXME: This essentially hardcodes the ResNet output dimension. Still better than hardcoding in-place.
            # TODO: Decide on some sizes in a principled way
            'training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            'augmentation_crop_size': (224, 224, 3),  # Size of a query/key input patch, yields at least 128 overlap
            'augmentation_gray_probability': 0.1,
            'augmentation_jitter_range': 0.2  # TODO: This is 0.4 in the original paper. Test with 0.4 (i.e. stronger)
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training data')
        try:
            training_dataset = self._load_dataset()
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Make sure the batch size and queue size are compatible
        assert self.parameters['moco_queue_size'] % self.parameters['batch_size'] == 0, \
            'Queue size must be a multiple of the batch size'

        self.log.info('Building models')
        model_backbone = self._construct_backbone(self.parameters['backbone'])
        backbone = rs.models.fastfcn_moco.FastFCNMoCoContrastBackbone(model_backbone)
        momentum_model_backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone = rs.models.fastfcn_moco.FastFCNMoCoContrastBackbone(momentum_model_backbone)

        momentum_backbone.trainable = False
        encoder, momentum_encoder = self._construct_heads(
            self.parameters['moco_head'],
            backbone,
            momentum_backbone
        )
        model = rs.models.moco.EncoderMoCoTrainingModel(
            encoder,
            momentum_encoder,
            momentum=self.parameters['moco_momentum'],
            temperature=self.parameters['moco_temperature'],
            queue_size=self.parameters['moco_queue_size'],
            features=self.parameters['moco_features']
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
        # TODO: More evaluation during training (e.g. some visualization techniques?)
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
        # FIXME: Could use target_tensors here to specify the (constant) targets instead of feeding them via dataset
        model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=learning_rate_schedule,
                momentum=self.parameters['momentum'],
                nesterov=self.parameters['nesterov']
            ),
            loss=losses,
            metrics=metrics
        )

        # TODO: Some callback which evaluates the representations each epoch?
        callbacks = [
                        self.keras.tensorboard_callback(),
                        self.keras.periodic_checkpoint_callback(
                            period=1,
                            checkpoint_template='{epoch:04d}-{loss:.4f}.h5'
                        ),
                        self._PeriodicBackboneCheckpointCallback(
                            model_backbone,
                            checkpoint_template='backbone_{epoch:04d}.h5',
                            log_dir=self.experiment_directory
                        )
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

        # First, crop a smaller area from the larger patch to ensure enough overlap
        dataset = dataset.map(lambda image: (tf.image.random_crop(image, self.parameters['training_image_size'])))

        # Then, apply the actual data augmentation, two times separately
        dataset = dataset.map(lambda image: (self._augment_sample(image), self._augment_sample(image)))

        # Add label and convert images to correct colour space
        # The target class is always 0, i.e. the positive keys are at index 0
        dataset = dataset.map(
            lambda image1, image2: (
                (rs.data.image.map_colorspace(image1), rs.data.image.map_colorspace(image2)),  # Images
                0  # Label
            )
        )

        # Batch samples
        # drop_remainder=True is crucial since the sample queue assumes queue size modulo batch size to be 0
        dataset = dataset.batch(self.parameters['batch_size'], drop_remainder=True)

        # Prefetch batches to decrease latency
        dataset = dataset.prefetch(self.parameters['prefetch_buffer_size'])

        return dataset

    def _augment_sample(self, image: tf.Tensor) -> tf.Tensor:
        # Random crop
        # TODO: Originally we would randomly crop and then rescale to desired size here
        cropped_sample = tf.image.random_crop(image, self.parameters['augmentation_crop_size'])

        # Randomly convert to grayscale
        grayscale_sample = rs.data.image.random_grayscale(
            cropped_sample,
            probability=self.parameters['augmentation_gray_probability']
        )

        # Random color jitter
        jitter_range = self.parameters['augmentation_jitter_range']
        jittered_sample = rs.data.image.random_color_jitter(
            grayscale_sample,
            jitter_range, jitter_range, jitter_range, jitter_range
        )

        # Random flip and rotation, this covers all possible permutations which do not require interpolation
        flipped_sample = tf.image.random_flip_left_right(jittered_sample)
        # flipped_sample = tf.image.random_flip_left_right(cropped_sample)
        rotated_sample = tf.image.rot90(
            flipped_sample,
            k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)  # Between 0 and 3 rotations
        )

        # TODO: There is some normalization according to (arXiv:1805.01978 [cs.CV]) happening at the end.
        #  However, those are some random constants whose origin I could not determine yet.
        normalized_sample = rotated_sample

        return normalized_sample

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        # FIXME: [v1] The original does shuffling batch norm across GPUs to avoid issues stemming from
        #  leaking statistics via then normalizatio.
        #  We need a solution which orks on a single GPU.
        #  arXiv:1905.09272 [cs.CV] does layer norm instead which is more suitable to our single-GPU case.
        #  This seems to fit similarly fast but we need to evaluate the effects on downstream performance.
        #  Furthermore, layer normalization requires much more memory than batch norm and thus reduces batch size etc.
        # TODO: [v1] try out (emulated) shuffling batch norm as well.
        #  This could be achieved by shuffling, splitting the batch and parallel batch norm layers.
        #  However, that might also be memory- and performance-inefficient.
        normalization_builder = rs.util.BatchNormalizationBuilder()

        # TODO: Just as a general reminder, we need to implement the improved ResNet version!

        kwargs = {
            'kernel_regularizer': tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
            'kernel_initializer': self.parameters['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

        resnet = None

        if name == 'ResNet50':
            resnet = rs.models.resnet.ResNet50Backbone(**kwargs)
        elif name == 'ResNet101':
            resnet = rs.models.resnet.ResNet101Backbone(**kwargs)
        else:
            raise AssertionError(f'Unexpected backbone name "{name}"')

        moco_backbone = rs.models.fastfcn_moco.FastFCNMoCoBackbone(
            resnet,
            kernel_initializer=self.parameters['kernel_initializer'],
            kernel_regularizer=tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
            jpu_features=self.parameters['jpu_features']
        )

        return moco_backbone

    def _construct_heads(
            self,
            head_type: str,
            backbone: tf.keras.Model,
            momentum_backbone: tf.keras.Model
    ) -> typing.Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        if head_type == 'fc':
            # v1 head
            encoder = rs.models.moco.FCHead(
                backbone,
                features=self.parameters['moco_features'],
                dense_initializer=self.parameters['dense_initializer'],
                kernel_regularizer=tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
                name='encoder'
            )
            momentum_encoder = rs.models.moco.FCHead(
                momentum_backbone,
                features=self.parameters['moco_features'],
                name='momentum_encoder',
                trainable=False
            )
        elif head_type == 'mlp':
            # v2 head
            encoder = rs.models.moco.MLPHead(
                backbone,
                output_features=self.parameters['moco_features'],
                intermediate_features=self.parameters['moco_mlp_features'],
                dense_initializer=self.parameters['dense_initializer'],
                kernel_regularizer=tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
                name='encoder'
            )
            momentum_encoder = rs.models.moco.MLPHead(
                momentum_backbone,
                output_features=self.parameters['moco_features'],
                intermediate_features=self.parameters['moco_mlp_features'],
                name='momentum_encoder',
                trainable=False
            )
        else:
            raise ValueError(f'Unexpected head type {head_type}')

        return encoder, momentum_encoder

    class _PeriodicBackboneCheckpointCallback(tf.keras.callbacks.Callback):
        def __init__(self, backbone, checkpoint_template: str, log_dir: str):
            super().__init__()
            self.backbone = backbone

            # Create checkpoint directory
            checkpoint_dir = os.path.join(log_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Create path template
            path_template = os.path.join(checkpoint_dir, checkpoint_template)
            self.path_template = path_template

        def on_epoch_end(self, epoch, logs=None):
            self.backbone.save_weights(self.path_template.format(epoch=epoch))


if __name__ == '__main__':
    main()
