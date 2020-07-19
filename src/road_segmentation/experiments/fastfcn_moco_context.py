import argparse
import logging
import typing

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN training with MoCo loss on semantic encodings instead of semantic loss.'
EXPERIMENT_TAG = 'fastfcn_moco_context'


def main():
    FastFCNMoCoContextExperiment().run()


class FastFCNMoCoContextExperiment(rs.framework.Experiment):
    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on the reference implementation at https://github.com/facebookresearch/moco
        # FIXME: This is the max fitting on a 1080Ti, original is 256
        parser.add_argument(
            '--moco-batch-size',
            type=int,
            default=32,
            help='Training batch size for contrastive loss on encodings.'
        )
        parser.add_argument(
            '--segmentation-batch-size',
            type=int,
            default=2,
            help='Training batch size for supervised segmentation loss.'
        )
        parser.add_argument('--learning-rate', type=float, default=3e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        # FIXME: What would be a sensible default?
        parser.add_argument('--prefetch-buffer-size', type=int, default=16, help='Number of batches to pre-fetch')
        parser.add_argument('--segmentation-loss-weight', type=float, default=1.0, help='Weight of segmentation loss')
        parser.add_argument('--moco-loss-weight', type=float, default=0.5, help='Weight of moco loss')
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )
        parser.add_argument(
            '--codewords',
            type=int,
            default=32,
            help='Number of codewords in the context encoding module'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'jpu_features': 512,  # FIXME: We could decrease those since we have less classes.
            'codewords': args.codewords,
            'backbone': args.backbone,
            'weight_decay': args.weight_decay,
            'segmentation_loss_weight': args.segmentation_loss_weight,
            'moco_loss_weight': args.moco_loss_weight,
            'head_dropout': 0.1,
            'output_upsampling': 'nearest',
            # FIXME: Keras uses glorot_uniform for both initializers.
            #  he_uniform is the same as the PyTorch default with an additional (justified) factor sqrt(6).
            #  Generally, there is no principled way to decide uniform vs normal.
            #  Also, He "should work better for ReLU" compared to Glorot but that is also not very clear.
            #  We should decide on which one to use.
            'kernel_initializer': 'he_uniform',
            'dense_initializer': 'he_uniform',
            'moco_batch_size': args.moco_batch_size,
            'segmentation_batch_size': args.segmentation_batch_size,
            'nesterov': True,
            'initial_learning_rate': args.learning_rate,
            # FIXME: The original authors decay to zero but small non-zero might be better
            'end_learning_rate': 1e-8,
            'learning_rate_decay': 0.9,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_temperature': 0.07,
            # Originally was 65536 (2^16). Decided that, in this context, 2048 is fine.
            'moco_queue_size': 2048,
            'se_loss_features': 2048,  # The context encoding module outputs this many features for the se loss
            'moco_mlp_features': 256,  # The MLP head outputs this number of features, which are used for contrasting
            # FIXME: This essentially hardcodes the ResNet output dimension. Still better than hardcoding in-place
            # TODO: Decide on some sizes in a principled way
            'moco_training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            # Size of a query/key input patch, yields at least 128 overlap
            'moco_augmentation_crop_size': (224, 224, 3),
            'moco_augmentation_gray_probability': 0.1,
            'moco_augmentation_max_relative_upsampling': 0.2,
            # TODO: This is 0.4 in the original paper. Test with 0.4 (i.e. stronger)
            'moco_augmentation_jitter_range': 0.2,
            'augmentation_interpolation': 'bilinear',
            # Scaling +-, output feature result in [384, 416]
            'segmentation_augmentation_max_relative_scaling': 0.04,
            'segmentation_augmentation_blur_probability': 0.5,
            'segmentation_augmentation_blur_size': 5,  # 5x5 Gaussian filter for blurring
            'segmentation_training_image_size': (384, 384)
        }

    def fit(self) -> typing.Any:
        try:
            training_paths, validation_paths = rs.data.cil.train_validation_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(training_paths)
            validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            self.log.debug(
                'Loaded %d training and %d validation samples',
                training_images.shape[0],
                validation_images.shape[0]
            )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            raise OSError('Unable to load data')
        self.log.info('Loading training data')
        try:
            training_dataset, validation_dataset = self._load_datasets(
                training_images,
                training_masks,
                validation_images,
                validation_masks
            )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        fastfcn = self._construct_fastfcn(self.parameters['backbone'])
        momentum_fastfcn = self._construct_fastfcn(self.parameters['backbone'])
        momentum_fastfcn.trainable = False

        encoder, momentum_encoder = self._construct_heads(
            fastfcn,
            momentum_fastfcn
        )

        steps_per_epoch = np.ceil(len(training_images) / self.parameters['segmentation_batch_size'])

        model = rs.models.fastfcn_moco_context.FastFCNMoCoContextTrainingModel(
            encoder=encoder,
            momentum_encoder=momentum_encoder,
            momentum=self.parameters['moco_momentum'],
            temperature=self.parameters['moco_temperature'],
            queue_size=self.parameters['moco_queue_size'],
            features=self.parameters['moco_mlp_features']
        )

        model.build(list(map(lambda spec: spec.shape, training_dataset.element_spec[0])))
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        learning_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.parameters['initial_learning_rate'],
            decay_steps=self.parameters['epochs'] * steps_per_epoch,
            end_learning_rate=self.parameters['end_learning_rate'],
            power=self.parameters['learning_rate_decay']
        )

        weight_deacy_factor = self.parameters['weight_decay'] / self.parameters['segmentation_initial_learning_rate']
        optimizer = tfa.optimizers.SGDW(
            weight_decay=lambda: weight_deacy_factor * learning_rate_scheduler(optimizer.iterations),
            learning_rate=learning_rate_scheduler,
            momentum=self.parameters['momentum'],
            nesterov=self.parameters['nesterov']
        )

        metrics = {
            'output_1': self.keras.default_metrics(threshold=0.0),
            'output_2': [
                tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='sparse_top_5_categorical_accuracy'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='sparse_top_1_categorical_accuracy')
            ]
        }

        losses = {
            'output_1': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'output_2': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        }

        loss_weights = {
            'output_1': self.parameters['segmentation_loss_weight'],
            'output_2': self.parameters['moco_loss_weight']
        }

        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

        log_predictions_callback = self.keras.log_predictions(
            validation_images=rs.data.image.rgb_to_cielab(validation_images),
            display_images=validation_images,
            prediction_idx=0,
            model=fastfcn
        )

        callbacks = [
                        self.keras.tensorboard_callback(),
                        self.keras.periodic_checkpoint_callback(
                            period=1,
                            checkpoint_template='{epoch:04d}-{loss:.4f}.h5'
                        ),
                        self.keras.best_checkpoint_callback(metric='val_output_1_binary_mean_f_score'),
                        log_predictions_callback
                    ] + model.create_callbacks()  # For MoCo updates

        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )

        return fastfcn

    def predict(
            self,
            classifier: typing.Any,
            images: typing.Dict[int, np.ndarray]
    ) -> typing.Dict[int, np.ndarray]:
        result = dict()
        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            image = rs.data.image.rgb_to_cielab(image)
            (raw_prediction,), _ = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)
            result[sample_id] = prediction

        return result

    def _load_datasets(
            self,
            training_images: np.ndarray,
            training_masks: np.ndarray,
            validation_images: np.ndarray,
            validation_masks: np.ndarray
    ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
        self.log.debug('Loading unlabelled data')
        # Uses all unsupervised samples in a flat order
        unlabelled_dataset = rs.data.unsupervised.shuffled_image_dataset(
            rs.data.unsupervised.processed_sample_paths(self.parameters['base_data_directory']),
            output_shape=(rs.data.unsupervised.PATCH_WIDTH, rs.data.unsupervised.PATCH_WIDTH, 3),
            seed=self.SEED
        )
        # First, augment the full sample and crop a smaller region out of it
        unlabelled_dataset = unlabelled_dataset.map(
            lambda image: self._moco_augment_full_sample(image),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Then, apply the actual data augmentation, two times separately
        unlabelled_dataset = unlabelled_dataset.map(
            lambda image: ((self._moco_augment_individual_patch(image), self._moco_augment_individual_patch(image)), 0),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Batch samples
        # drop_remainder=True is crucial since the sample queue assumes queue size modulo batch size to be 0
        unlabelled_dataset = unlabelled_dataset.batch(self.parameters['moco_batch_size'], drop_remainder=True)
        # TODO: Maybe prefetch on `unlabelled_dataset` here to prevent bottleneck

        labelled_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        labelled_dataset = labelled_dataset.shuffle(buffer_size=training_images.shape[0])
        labelled_dataset = labelled_dataset.map(lambda image, mask: self._fastfcn_augment_sample(image, mask))
        labelled_dataset = labelled_dataset.batch(self.parameters['segmentation_batch_size'])

        training_dataset = tf.data.Dataset.zip((unlabelled_dataset, labelled_dataset))

        # Tuples are structured as follows:
        # unlabelled: ((unlabelled image augmented, unlabelled image augmented (differently)), contrastive loss label)
        # labelled: (labelled image, segmentation mask)
        # Result is a single training batch (e.g. every tensor in the tuple is a batch):
        # ((labelled input image, unlabelled image augmented, unlabelled image augemnted),
        #   (segmentation mask label, contrastive loss label)
        # )
        training_dataset = training_dataset.map(
            lambda unlabelled, labelled: (
                (labelled[0], unlabelled[0][0], unlabelled[0][1]), (labelled[1], unlabelled[1])
            )
        )

        # Prefetch batches to decrease latency
        training_dataset = training_dataset.prefetch(self.parameters['prefetch_buffer_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Validation images can be directly converted to the model colour space
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(validation_images), validation_masks)
        )

        zeros_image = tf.zeros((1,) + self.parameters['moco_augmentation_crop_size'], dtype=tf.float32)
        zeros_dataset = tf.data.Dataset.from_tensor_slices((zeros_image, [0]))
        zeros_dataset = zeros_dataset.repeat(len(validation_images))
        validation_dataset = tf.data.Dataset.zip((zeros_dataset, validation_dataset))

        # Tuples are structured as follows:
        # zeros: (zero placeholder image, zero placeholder label)
        # labelled: (labelled image, segmentation mask)
        # Since evaluation the contrastive loss is not useful, we use dummy inputs.
        # Result is a single evaluation batch:
        # ((labelled input image, zero placeholder image, zero placeholder image),
        #   (segmentation mask label, contrastive loss label)
        # )
        validation_dataset = validation_dataset.map(
            lambda zeros, labelled: ((labelled[0], zeros[0], zeros[0]), (labelled[1], zeros[1]))
        )
        # validation_dataset = validation_dataset.map(
        #     lambda zeros, labelled: ((labelled[0], zeros[0], zeros[0]), (labelled[1], zeros[1]))
        # )
        validation_dataset = validation_dataset.batch(1)

        return training_dataset, validation_dataset

    def _moco_augment_full_sample(self, image: tf.Tensor) -> tf.Tensor:

        # Random upsampling
        upsampling_factor = tf.random.uniform(
            shape=[],
            minval=1.0,
            maxval=1.0 + self.parameters['moco_augmentation_max_relative_upsampling']
        )
        input_height, input_width, input_channels = tf.unstack(tf.shape(image))
        input_height, input_width = tf.unstack(tf.cast((input_height, input_width), dtype=tf.float32))
        scaled_size = tf.cast(
            tf.round((input_height * upsampling_factor, input_width * upsampling_factor)),
            tf.int32
        )

        upsampled_image = tf.image.resize(image, scaled_size, method=self.parameters['augmentation_interpolation'])

        # Then, random rotate and crop a smaller range from the image
        cropped_sample = rs.data.image.random_rotate_and_crop(
            upsampled_image,
            self.parameters['moco_training_image_size'][0]
        )

        return cropped_sample

    def _moco_augment_individual_patch(self, image: tf.Tensor) -> tf.Tensor:

        flipped_sample = tf.image.random_flip_left_right(image)

        cropped_image = rs.data.image.random_rotate_and_crop(
            flipped_sample,
            self.parameters['moco_augmentation_crop_size'][0]
        )

        # Randomly convert to grayscale
        grayscale_sample = rs.data.image.random_grayscale(
            cropped_image,
            probability=self.parameters['moco_augmentation_gray_probability']
        )

        # Random color jitter
        jitter_range = self.parameters['moco_augmentation_jitter_range']
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

    def _fastfcn_augment_sample(self, image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        # Random Gaussian blurring
        do_blur = tf.random.uniform(shape=[], dtype=tf.float32) < self.parameters[
            'segmentation_augmentation_blur_probability']
        blurred_image = tf.cond(do_blur, lambda: self._augment_blur(image), lambda: image)
        blurred_image.set_shape(image.shape)  # Must set shape manually since it cannot be inferred from tf.cond

        # Random scaling
        scaling_factor = tf.random.uniform(
            shape=[],
            minval=1.0 - self.parameters['segmentation_augmentation_max_relative_scaling'],
            maxval=1.0 + self.parameters['segmentation_augmentation_max_relative_scaling']
        )
        input_height, input_width, _ = tf.unstack(tf.cast(tf.shape(blurred_image), tf.float32))
        scaled_size = tf.cast(
            tf.round((input_height * scaling_factor, input_width * scaling_factor)),
            tf.int32
        )
        scaled_image = tf.image.resize(blurred_image, scaled_size, method=self.parameters['augmentation_interpolation'])
        scaled_mask = tf.image.resize(mask, scaled_size, method='nearest')

        # Combine image and mask to ensure same transformations are applied
        concatenated_sample = tf.concat((scaled_image, scaled_mask), axis=-1)

        # Random flip and rotation, this covers all possible permutations which do not require interpolation
        flipped_sample = tf.image.random_flip_left_right(concatenated_sample)
        num_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        rotated_sample = tf.image.rot90(flipped_sample, num_rotations)

        # Random crop
        crop_size = self.parameters['segmentation_training_image_size'] + (4,)  # 3 colour channels + 1 mask channel
        cropped_sample = tf.image.random_crop(rotated_sample, crop_size)

        # Split combined image and mask again
        output_image = cropped_sample[:, :, :3]
        output_mask = cropped_sample[:, :, 3:]

        # Convert mask to labels in {0, 1} but keep as floats
        output_mask = tf.round(output_mask)

        # Convert image to CIE Lab
        # This has to be done after the other transformations since some assume RGB inputs
        output_image_lab = rs.data.image.map_colorspace(output_image)

        # FIXME: It would make sense to apply colour shifts but the original paper does not

        return output_image_lab, output_mask

    def _augment_blur(self, image: tf.Tensor) -> tf.Tensor:
        # Pick standard deviation randomly in [0.5, 1)
        sigma = tf.random.uniform(shape=[], minval=0.5, maxval=1.0, dtype=tf.float32)
        sigma_squared = tf.square(sigma)

        # FIXME: This would be significantly faster if applied as two 1D convolutions instead of a 2D one

        # Calculate Gaussian filter kernel
        kernel_size = self.parameters['segmentation_augmentation_blur_size']
        half_kernel_size = kernel_size // 2
        grid_y_squared, grid_x_squared = np.square(
            np.mgrid[-half_kernel_size:half_kernel_size + 1, -half_kernel_size:half_kernel_size + 1]
        )
        coordinates = grid_y_squared + grid_x_squared
        kernel = 1.0 / (2.0 * np.pi * sigma_squared) * tf.exp(
            - coordinates / (2.0 * sigma_squared)
        )
        kernel = tf.reshape(kernel, (kernel_size, kernel_size, 1, 1))
        kernel = tf.repeat(kernel, 3, axis=2)
        # => Kernel shape is [kernel_size, kernel_size, 3, 1]

        # Pad image using reflection padding (not available in depthwise_conv2d)
        padded_image = tf.pad(
            image,
            paddings=((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
            mode='REFLECT'
        )
        padded_image = tf.expand_dims(padded_image, axis=0)

        # Finally apply Gaussian filter
        blurred_image = tf.nn.depthwise_conv2d(padded_image, kernel, strides=(1, 1, 1, 1), padding='VALID')

        # Result might have values outside the normalized range, clip those
        output = tf.clip_by_value(blurred_image[0], 0.0, 1.0)
        return output

    def _construct_fastfcn(self, name: str) -> tf.keras.Model:
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

        resnet_kwargs = {
            'kernel_initializer': self.parameters['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

        # TODO: Just as a general reminder, we need to implement the improved ResNet version!

        backbone = None
        if name == 'ResNet50':
            backbone = rs.models.resnet.ResNet50Backbone(**resnet_kwargs)
        if name == 'ResNet101':
            backbone = rs.models.resnet.ResNet101Backbone(**resnet_kwargs)

        if backbone is None:
            raise AssertionError(f'Unexpected backbone name "{name}"')

        fastfcn = rs.models.fastfcn.FastFCN(
            backbone,
            jpu_features=self.parameters['jpu_features'],
            se_loss_features=self.parameters['se_loss_features'],
            head_dropout_rate=self.parameters['head_dropout'],
            dense_initializer=self.parameters['dense_initializer'],
            output_upsampling=self.parameters['output_upsampling'],
            kernel_initializer=self.parameters['kernel_initializer'],
            codewords=self.parameters['codewords']
        )
        return fastfcn

    def _construct_heads(
            self,
            fastfcn: tf.keras.Model,
            momentum_fastfcn: tf.keras.Model
    ) -> typing.Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        # MoCo v2 head
        encoder = rs.models.fastfcn_moco_context.FastFCNMoCoContextMLPHead(
            fastfcn,
            output_features=self.parameters['moco_mlp_features'],
            dense_initializer=self.parameters['dense_initializer'],
            name='encoder'
        )
        momentum_encoder = rs.models.fastfcn_moco_context.FastFCNMoCoContextMLPHead(
            momentum_fastfcn,
            output_features=self.parameters['moco_mlp_features'],
            name='momentum_encoder',
            trainable=False
        )

        return encoder, momentum_encoder


if __name__ == '__main__':
    main()
