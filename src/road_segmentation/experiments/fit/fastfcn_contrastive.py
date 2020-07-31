import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN with contrastive context encoding module loss'
EXPERIMENT_TAG = 'fastfcn_contrastive'


def main():
    FastFCNContrastiveExperiment().run()


class FastFCNContrastiveExperiment(rs.framework.FitExperiment):
    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on the reference implementation at https://github.com/facebookresearch/moco
        parser.add_argument(
            '--contrastive-batch-size',
            type=int,
            default=4,
            help='Training batch size for contrastive loss.'
        )
        parser.add_argument(
            '--segmentation-batch-size',
            type=int,
            default=4,
            help='Training batch size for supervised segmentation loss'
        )
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
            'codewords': 32,
            'backbone': args.backbone,
            'weight_decay': 1e-4,
            'segmentation_loss_ratio': 0.8,
            'head_dropout': 0.1,
            'kernel_initializer': 'he_uniform',
            'dense_initializer': 'he_uniform',
            'moco_batch_size': args.contrastive_batch_size,
            'segmentation_batch_size': args.segmentation_batch_size,
            'initial_learning_rate': 3e-2,
            'end_learning_rate': 1e-8,
            'learning_rate_decay': 0.9,
            'momentum': 0.9,
            'epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_momentum': 0.999,
            'moco_initial_temperature': 1e1,
            'moco_min_temperature': 1e-5,
            'moco_temperature_decay': 0.99,
            # Originally was 65536 (2^16). Decided that, in this context, 2048 is fine.
            'moco_queue_size': 2048,
            'se_loss_features': 2048,  # The context encoding module outputs this many features for the se loss
            'moco_mlp_features': 256,  # The MLP head outputs this number of features, which are used for contrasting
            'moco_training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            # Size of a query/key input patch, yields at least 128 overlap
            'moco_augmentation_crop_size': (224, 224, 3),
            # Scaling +-, output feature result in [384, 416]
            'segmentation_augmentation_max_relative_scaling': 0.04,
            'segmentation_training_image_size': (384, 384)
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training data')
        try:
            training_paths = rs.data.cil.training_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(training_paths)
            self.log.debug('Loaded %d labelled samples', training_images.shape[0])

            training_dataset = self._load_datasets(training_images, training_masks)
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            raise
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        fastfcn = self._construct_fastfcn(self.parameters['backbone'])
        momentum_fastfcn = self._construct_fastfcn(self.parameters['backbone'])
        momentum_fastfcn.trainable = False

        encoder, momentum_encoder = self._construct_heads(
            fastfcn,
            momentum_fastfcn
        )

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

        steps_per_epoch = np.ceil(len(training_images) / self.parameters['segmentation_batch_size'])
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

        loss_weights = {
            'output_1': self.parameters['segmentation_loss_ratio'],
            'output_2': 1.0 - self.parameters['segmentation_loss_ratio']
        }

        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(checkpoint_template='{epoch:04d}-{loss:.4f}.h5'),
            self.keras.best_checkpoint_callback(metric='output_1_binary_mean_accuracy'),
            self.keras.log_learning_rate_callback(),
            self.keras.decay_temperature_callback(
                initial_temperature=self.parameters['moco_initial_temperature'],
                min_temperature=self.parameters['moco_min_temperature'],
                decay_rate=self.parameters['moco_temperature_decay']
            ),
            self.keras.log_temperature_callback()
        ] + model.create_callbacks()  # For MoCo updates

        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            callbacks=callbacks
        )

        return fastfcn

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            # Convert to model colour space
            image = rs.data.image.rgb_to_cielab(image)

            # Predict labels at model's output stride
            raw_prediction, _ = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1.0, 0.0)

            # Threshold patches to create final prediction
            prediction = rs.data.cil.segmentation_to_patch_labels(prediction, rs.models.fastfcn.OUTPUT_STRIDE)[0]
            prediction = prediction.astype(int)

            result[sample_id] = prediction

        return result

    def _load_datasets(
            self,
            training_images: np.ndarray,
            training_masks: np.ndarray
    ) -> tf.data.Dataset:
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

        labelled_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        labelled_dataset = labelled_dataset.shuffle(buffer_size=training_images.shape[0])
        labelled_dataset = labelled_dataset.map(lambda image, mask: rs.data.cil.augment_image(
            image,
            mask,
            crop_size=self.parameters['segmentation_training_image_size'],
            max_relative_scaling=self.parameters['segmentation_augmentation_max_relative_scaling'],
            model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE
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

        return training_dataset

    def _construct_fastfcn(self, name: str) -> tf.keras.Model:
        # The original does shuffling batch norm across GPUs to avoid issues stemming from
        #  leaking statistics via the normalization.
        #  We need a solution which works on a single GPU.
        #  arXiv:1905.09272 [cs.CV] does layer norm instead which is more suitable to our single-GPU case.
        #  This seems to fit similarly fast but we need to evaluate the effects on downstream performance.
        #  Furthermore, layer normalization requires much more memory than batch norm and thus reduces batch size etc.
        normalization_builder = rs.util.LayerNormalizationBuilder()

        resnet_kwargs = {
            'kernel_initializer': self.parameters['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

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
            head_dropout_rate=self.parameters['head_dropout'],
            kernel_initializer=self.parameters['kernel_initializer'],
            dense_initializer=self.parameters['dense_initializer'],
            se_loss_features=self.parameters['se_loss_features'],
            codewords=self.parameters['codewords'],
            kernel_regularizer=None
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
