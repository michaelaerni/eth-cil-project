import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'MoCo Spatial Representations Only'
EXPERIMENT_TAG = 'moco_spatial_representations'


def main():
    MoCoSpatialRepresentationsExperiment().run()


class MoCoSpatialRepresentationsExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on the reference implementation at https://github.com/facebookresearch/moco
        parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')  # FIXME: This is the max fitting on a 1080Ti, original is 256
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
        parser.add_argument(
            '--moco-head',
            type=str,
            default='fc',  # TODO: Experiment with different heads
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
            'batch_size': args.batch_size,
            'nesterov': True,
            'initial_learning_rate': args.learning_rate,
            'learning_rate_schedule': (120, 160),  # TODO: Test different schedules
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_features': 32,
            'moco_features_size': 4,  # 4x4 rectangle of features is being cropped (resulting from 128x128 overlap)
            'moco_temperature': 0.07,
            'moco_queue_size': 16384,  # TODO: Originally was 65536 (2^16)
            'moco_head': args.moco_head,
            'moco_mlp_features': 2048,  # FIXME: This essentially hardcodes the ResNet output dimension. Still better than hardcoding in-place.
            # TODO: Decide on some sizes in a principled way
            'training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            'augmentation_crop_size': (224, 224, 3),  # Size of a query/key input patch, yields at least 128 overlap
            'augmentation_gray_probability': 0.1,
            'augmentation_jitter_range': 0.2,  # TODO: This is 0.4 in the original paper. Test with 0.4 (i.e. stronger)
            'augmentation_alignment_stride': 32
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
        assert self.parameters['moco_queue_size'] % self.parameters['batch_size'] == 0,\
            'Queue size must be a multiple of the batch size'

        self.log.info('Building models')
        backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone = self._construct_backbone(self.parameters['backbone'])
        momentum_backbone.trainable = False
        encoder, momentum_encoder = self._construct_heads(
            self.parameters['moco_head'],
            backbone,
            momentum_backbone
        )
        model = rs.models.moco.SpatialEncoderMoCoTrainingModel(
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
            self.keras.periodic_checkpoint_callback(period=1, checkpoint_template='{epoch:04d}-{loss:.4f}.h5')
        ] + model.create_callbacks()  # Required MoCo updates

        # Fit model
        self.log.info('Fitting model')
        # TODO: Check GPU utilization
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
            output_shape=(rs.data.unsupervised.PATCH_WIDTH, rs.data.unsupervised.PATCH_WIDTH, 3),
            seed=self.SEED
        )
        # First, augment the full sample and crop a smaller region out of it
        dataset = dataset.map(lambda image: self._augment_full_sample(image))

        # Second, crop query/key and perform reversible transformations
        dataset = dataset.map(lambda image: self._create_query_key(image))

        # Then, apply the actual data augmentation, two times separately
        dataset = dataset.map(
            lambda query, key, aug: (self._augment_individual_patch(query), self._augment_individual_patch(key)) + aug
        )

        # Batch samples
        # drop_remainder=True is crucial since the sample queue assumes queue size modulo batch size to be 0
        dataset = dataset.batch(self.parameters['batch_size'], drop_remainder=True)

        # TODO: This is a bit hacky
        # Finally, the labels have a different size compared to the batch and are thus added here.
        # This is required for all "normal" metrics and losses to work correctly.
        dataset = dataset.map(lambda *sample: (sample, np.zeros(
            (self.parameters['batch_size'] * (self.parameters['moco_features_size'] ** 2)),
            dtype=np.int
        )))

        # Prefetch batches to decrease latency
        dataset = dataset.prefetch(self.parameters['prefetch_buffer_size'])

        return dataset

    def _augment_full_sample(self, image: tf.Tensor) -> tf.Tensor:
        # First, randomly flip left and right
        flipped_sample = tf.image.random_flip_left_right(image)

        # TODO: Also need to scale up and down randomly here!

        # Then, random rotate and crop a smaller range from the image
        cropped_sample = rs.data.image.random_rotate_and_crop(flipped_sample, self.parameters['training_image_size'][0])

        return cropped_sample

    def _create_query_key(self, full_image: tf.Tensor) -> typing.Tuple[
        tf.Tensor,
        tf.Tensor,
        typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
    ]:
        # Returns (query, key, augmentations) where augmentations are
        # (query_offset_x, query_offset_y, key_offset_x, key_offset_y, is_flipped, rotations).
        # All offsets are integers relative to the target stride.
        # Flip and rotation are only applied to the key image since the full image is already augmented.

        target_stride = self.parameters['augmentation_alignment_stride']

        # Determine range for random offset in target stride
        # This assumes square images everywhere
        input_size_pixels, _, _ = self.parameters['training_image_size']
        output_size_pixels, _, _ = self.parameters['augmentation_crop_size']
        offset_range_stride = (input_size_pixels - output_size_pixels) // target_stride

        # Determine and perform random crops (in stride) for both query and key
        query_offset_x, query_offset_y = tf.unstack(
            tf.random.uniform([2], minval=0, maxval=offset_range_stride, dtype=tf.int32)
        )
        query_offset_x_pixel, query_offset_y_pixel = query_offset_x * target_stride, query_offset_y * target_stride
        query = full_image[
            query_offset_y_pixel:query_offset_y_pixel+output_size_pixels,
            query_offset_x_pixel:query_offset_x_pixel+output_size_pixels,
            :
        ]
        key_offset_x, key_offset_y = tf.unstack(
            tf.random.uniform([2], minval=0, maxval=offset_range_stride, dtype=tf.int32)
        )
        key_offset_x_pixel, key_offset_y_pixel = key_offset_x * target_stride, key_offset_y * target_stride
        key_cut = full_image[
            key_offset_y_pixel:key_offset_y_pixel+output_size_pixels,
            key_offset_x_pixel:key_offset_x_pixel+output_size_pixels,
            :
        ]

        # Restore shape information
        query.set_shape(self.parameters['augmentation_crop_size'])
        key_cut.set_shape(self.parameters['augmentation_crop_size'])

        # Determine and perform random horizontal flips
        do_flip = tf.random.uniform([], dtype=tf.float32) < 0.5
        key_flipped = tf.cond(do_flip, lambda: tf.image.flip_left_right(key_cut), lambda: key_cut)
        key_flipped.set_shape(key_cut.get_shape())

        # Determine and perform random rotations
        rotations = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        key_rotated = tf.image.rot90(key_flipped, k=rotations)

        key = key_rotated

        return query, key, (
            query_offset_x,
            query_offset_y,
            key_offset_x,
            key_offset_y,
            do_flip,
            rotations
        )

    def _augment_individual_patch(self, image: tf.Tensor) -> tf.Tensor:
        # TODO: Here we can randomly upscale the image (such that it remains within a single stride)
        upsampled_sample = image

        # Randomly convert to grayscale
        grayscale_sample = rs.data.image.random_grayscale(
            upsampled_sample,
            probability=self.parameters['augmentation_gray_probability']
        )

        # Random color jitter
        jitter_range = self.parameters['augmentation_jitter_range']
        jittered_sample = rs.data.image.random_color_jitter(
            grayscale_sample,
            jitter_range, jitter_range, jitter_range, jitter_range
        )

        # TODO: There is some normalization according to (arXiv:1805.01978 [cs.CV]) happening at the end.
        #  However, those are some random constants whose origin I could not determine yet.
        normalized_sample = jittered_sample

        # Finally, convert to target colorspace
        output_image = rs.data.image.map_colorspace(normalized_sample)

        return output_image

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        # FIXME: [v1] The original does shuffling batch norm across GPUs to avoid issues stemming from
        #  leaking statistics via the normalization.
        #  We need a solution which works on a single GPU.
        #  arXiv:1905.09272 [cs.CV] does layer norm instead which is more suitable to our single-GPU case.
        #  This seems to fit similarly fast but we need to evaluate the effects on downstream performance.
        #  Furthermore, layer normalization requires much more memory than batch norm and thus reduces batch size etc.
        # TODO: [v1] try out (emulated) shuffling batch norm as well.
        #  This could be achieved by shuffling, splitting the batch and parallel batch norm layers.
        #  However, that might also be memory- and performance-inefficient.
        normalization_builder = rs.util.LayerNormalizationBuilder()

        kwargs = {
            'weight_decay': self.parameters['weight_decay'],
            'kernel_initializer': self.parameters['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

        # TODO: Just as a general reminder, we need to implement the improved ResNet version!

        if name == 'ResNet50':
            return rs.models.resnet.ResNet50Backbone(**kwargs)
        if name == 'ResNet101':
            return rs.models.resnet.ResNet101Backbone(**kwargs)

        raise AssertionError(f'Unexpected backbone name "{name}"')

    def _construct_heads(
            self,
            head_type: str,
            backbone: tf.keras.Model,
            momentum_backbone: tf.keras.Model
    ) -> typing.Tuple[rs.models.moco.Base2DHead, rs.models.moco.Base2DHead]:
        if head_type == 'fc':
            # Fully connected head
            encoder = rs.models.moco.FC2DHead(
                backbone,
                features=self.parameters['moco_features'],
                feature_rectangle_size=self.parameters['moco_features_size'],
                undo_spatial_transformations=False,
                dense_initializer=self.parameters['dense_initializer'],
                weight_decay=self.parameters['weight_decay'],
                name='encoder'
            )
            momentum_encoder = rs.models.moco.FC2DHead(
                momentum_backbone,
                features=self.parameters['moco_features'],
                feature_rectangle_size=self.parameters['moco_features_size'],
                undo_spatial_transformations=True,
                name='momentum_encoder',
                trainable=False
            )
        elif head_type == 'mlp':
            raise NotImplementedError('MLP 2D head not implemented yet')
        else:
            raise ValueError(f'Unexpected head type {head_type}')

        return encoder, momentum_encoder


if __name__ == '__main__':
    main()
