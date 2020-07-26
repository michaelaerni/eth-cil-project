import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN training with MoCo loss on semantic encodings instead of semantic loss.'
EXPERIMENT_TAG = 'fastfcn_moco_context'


def main():
    FastFCNMoCoContextExperiment().run()


class FastFCNMoCoContextExperiment(rs.framework.FitExperiment):
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
            default=4,
            help='Training batch size for contrastive loss on encodings.'
        )
        parser.add_argument(
            '--segmentation-batch-size',
            type=int,
            default=4,
            help='Training batch size for supervised segmentation loss.'
        )
        parser.add_argument('--learning-rate', type=float, default=3e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
        # FIXME: What would be a sensible default?
        parser.add_argument('--prefetch-buffer-size', type=int, default=16, help='Number of batches to pre-fetch')
        parser.add_argument('--segmentation-loss-weight', type=float, default=0.8, help='Weight of segmentation loss')
        parser.add_argument('--moco-loss-weight', type=float, default=0.2, help='Weight of moco loss')
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
        ),
        parser.add_argument('--moco-init-temperature', type=float, default=0.07, help='Init temperature for moco loss'),
        parser.add_argument(
            '--moco-min-temperature',
            type=float,
            default=1.0,
            help='Minimum temperature for moco loss'
        )
        parser.add_argument(
            '--moco-temperature-decay',
            type=str,
            default='none',
            choices=['none', 'exponential'],
            help='Which temperature decay is applied'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        # TODO: Adjust after search
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
            'initial_learning_rate': args.learning_rate,
            # FIXME: The original authors decay to zero but small non-zero might be better
            'end_learning_rate': 1e-8,
            'learning_rate_decay': 0.9,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_initial_temperature': args.moco_init_temperature,
            'moco_min_temperature': args.moco_min_temperature,
            'moco_temperature_decay': args.moco_temperature_decay,
            # Originally was 65536 (2^16). Decided that, in this context, 2048 is fine.
            'moco_queue_size': 2048,
            'se_loss_features': 2048,  # The context encoding module outputs this many features for the se loss
            'moco_mlp_features': 256,  # The MLP head outputs this number of features, which are used for contrasting
            # FIXME: This essentially hardcodes the ResNet output dimension. Still better than hardcoding in-place
            # TODO: Decide on some sizes in a principled way
            'moco_training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            # Size of a query/key input patch, yields at least 128 overlap
            'moco_augmentation_crop_size': (224, 224, 3),
            # Scaling +-, output feature result in [384, 416]
            'segmentation_augmentation_max_relative_scaling': 0.04,
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
            temperature=self.parameters['moco_initial_temperature'],
            queue_size=self.parameters['moco_queue_size'],
            features=self.parameters['moco_mlp_features']
        )

        model.build(list(map(lambda spec: spec.shape, training_dataset.element_spec[0])))
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        optimizer = self.keras.build_optimizer(
            total_steps=self.parameters['epochs'] * steps_per_epoch,
            initial_learning_rate=self.parameters['initial_learning_rate'],
            end_learning_rate=self.parameters['end_learning_rate'],
            learning_rate_decay=self.parameters['learning_rate_decay'],
            momentum=self.parameters['momentum'],
            weight_decay=self.parameters['weight_decay']
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

        # Floating point comparison checks if the loss weights sum up to approximately one
        if abs((self.parameters['segmentation_loss_weight'] + self.parameters['moco_loss_weight']) - 1.0) > 1e-9:
            raise ValueError(
                "Sum {} + {} = {} but should equal to 1.0".format(
                    self.parameters['segmentation_loss_weight'],
                    self.parameters['moco_loss_weight'],
                    self.parameters['segmentation_loss_weight'] + self.parameters['moco_loss_weight']
                )
            )

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
            fixed_model=fastfcn
        )

        callbacks = [
                        self.keras.tensorboard_callback(),
                        self.keras.periodic_checkpoint_callback(
                            checkpoint_template='{epoch:04d}-{loss:.4f}.h5'
                        ),
                        self.keras.best_checkpoint_callback(metric='val_output_1_binary_mean_f_score'),
                        log_predictions_callback,
                        self.keras.log_learning_rate_callback()
                    ] + model.create_callbacks()  # For MoCo updates

        if self.parameters['moco_temperature_decay'] == 'exponential':
            callbacks.append(
                self.keras.decay_temperature_callback(
                    initial_temperature=self.parameters['moco_initial_temperature'],
                    min_temperature=self.parameters['moco_min_temperature'],
                    decay_steps=self.parameters['epochs'],
                    decay_rate=None
                )
            )

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
            rs.data.unsupervised.processed_sample_paths(self.data_directory),
            output_shape=(rs.data.unsupervised.PATCH_WIDTH, rs.data.unsupervised.PATCH_WIDTH, 3),
            seed=self.SEED
        )
        # First, augment the full sample and crop a smaller region out of it
        unlabelled_dataset = unlabelled_dataset.map(
            lambda image: rs.data.unsupervised.augment_full_sample(image, self.parameters['moco_training_image_size']),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Then, apply the actual data augmentation, two times separately
        unlabelled_dataset = unlabelled_dataset.map(
            lambda image: ((
                rs.data.unsupervised.augment_patch(image, crop_size=self.parameters['moco_augmentation_crop_size']),
                rs.data.unsupervised.augment_patch(image, crop_size=self.parameters['moco_augmentation_crop_size'])
            ), 0),
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        # Batch samples
        # drop_remainder=True is crucial since the sample queue assumes queue size modulo batch size to be 0
        unlabelled_dataset = unlabelled_dataset.batch(self.parameters['moco_batch_size'], drop_remainder=True)
        # TODO: Maybe prefetch on `unlabelled_dataset` here to prevent bottleneck

        labelled_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        labelled_dataset = labelled_dataset.shuffle(buffer_size=training_images.shape[0])
        labelled_dataset = labelled_dataset.map(lambda image, mask: rs.data.cil.augment_image(
            image,
            mask,
            max_relative_scaling=self.parameters['segmentation_augmentation_max_relative_scaling'],
            crop_size=self.parameters['segmentation_training_image_size']
        ))
        labelled_dataset = labelled_dataset.batch(self.parameters['segmentation_batch_size'])

        training_dataset = tf.data.Dataset.zip((unlabelled_dataset, labelled_dataset))

        # Tuples are structured as follows:
        # unlabelled: ((unlabelled image augmented, unlabelled image augmented (differently)), contrastive loss label)
        # labelled: (labelled image, segmentation mask)
        # Result is a single training batch (e.g. every tensor in the tuple is a batch):
        # ((labelled input image, unlabelled image augmented, unlabelled image augmented),
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
