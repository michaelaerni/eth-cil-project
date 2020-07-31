import argparse
import typing

import ax
import numpy as np
import sklearn.model_selection
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN with contrastive context encoding module loss parameter search'
EXPERIMENT_TAG = 'fastfcn_contrastive_search'


def main():
    FastFCNContrastiveSearchExperiment().run()


class FastFCNContrastiveSearchExperiment(rs.framework.SearchExperiment):
    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on the reference implementation at https://github.com/facebookresearch/moco
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

        return {
            'backbone': args.backbone,
            'moco_batch_size': args.moco_batch_size,
            'segmentation_batch_size': args.segmentation_batch_size,
            'max_epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'moco_training_image_size': (320, 320, 3),  # Initial crop size before augmentation and splitting
            # Size of a query/key input patch, yields at least 128 overlap
            'moco_augmentation_crop_size': (224, 224, 3),
            # Scaling +-, output feature result in [384, 416]
            'segmentation_augmentation_max_relative_scaling': 0.04,
            'segmentation_training_image_size': (384, 384)
        }

    def build_search_space(self) -> ax.SearchSpace:
        parameters = [
            ax.FixedParameter('jpu_features', ax.ParameterType.INT, value=512),
            ax.FixedParameter('max_epochs', ax.ParameterType.INT, value=self.parameters['max_epochs']),
            ax.FixedParameter('backbone', ax.ParameterType.STRING, value=self.parameters['backbone']),
            ax.FixedParameter('weight_decay', ax.ParameterType.FLOAT, value=1e-4),
            ax.RangeParameter('head_dropout', ax.ParameterType.FLOAT, lower=0.0, upper=0.5),
            ax.RangeParameter('segmentation_loss_ratio', ax.ParameterType.FLOAT, lower=0.0, upper=1.0),
            ax.FixedParameter('kernel_initializer', ax.ParameterType.STRING, value='he_normal'),
            ax.FixedParameter('dense_initializer', ax.ParameterType.STRING, value='he_uniform'),
            # Only for the dense weights in the Encoder head
            ax.RangeParameter('initial_learning_rate_exp', ax.ParameterType.FLOAT, lower=-5.0, upper=0.0),
            ax.FixedParameter('end_learning_rate_exp', ax.ParameterType.FLOAT, value=-8.0),
            ax.RangeParameter(
                'learning_rate_decay',
                ax.ParameterType.FLOAT,
                lower=np.finfo(float).eps,
                upper=1.0,
                log_scale=True
            ),
            ax.FixedParameter('momentum', ax.ParameterType.FLOAT, value=0.9),
            ax.FixedParameter('moco_batch_size', ax.ParameterType.INT, value=self.parameters['moco_batch_size']),
            ax.FixedParameter(
                'segmentation_batch_size',
                ax.ParameterType.INT,
                value=self.parameters['segmentation_batch_size']
            ),
            ax.RangeParameter('moco_momentum', ax.ParameterType.FLOAT, lower=0.9, upper=1.0, log_scale=True),
            ax.RangeParameter(
                'moco_initial_temperature',
                ax.ParameterType.FLOAT,
                lower=1e-3,
                upper=20.,
                log_scale=True
            ),
            ax.FixedParameter('moco_min_temperature', ax.ParameterType.FLOAT, value=1e-5),
            ax.RangeParameter('moco_temperature_decay', ax.ParameterType.FLOAT, lower=0.9, upper=1.0, log_scale=True),
            ax.FixedParameter('moco_queue_size', ax.ParameterType.INT, value=2048),
            ax.FixedParameter('se_loss_features', ax.ParameterType.INT, value=2048),
            # The context encoding module outputs this many features for the se loss
            ax.FixedParameter('moco_mlp_features', ax.ParameterType.INT, value=256),
            # The MLP head outputs this number of features, which are used for contrasting
        ]

        return ax.SearchSpace(parameters)

    def run_fold(
            self,
            parameterization: typing.Dict[str, typing.Union[float, str, bool, int]],
            supervised_training_images: np.ndarray,
            supervised_training_masks: np.ndarray,
            supervised_validation_images: np.ndarray,
            supervised_validation_masks: np.ndarray,
            unsupervised_training_sample_paths: np.ndarray,
            unsupervised_validation_sample_paths: np.ndarray) -> float:

        training_dataset, validation_dataset, validation_dataset_large = self._load_datasets(
            supervised_training_images,
            supervised_training_masks,
            supervised_validation_images,
            supervised_validation_masks,
            unsupervised_training_sample_paths
        )

        fastfcn = self._construct_fastfcn(parameterization)
        momentum_fastfcn = self._construct_fastfcn(parameterization)
        momentum_fastfcn.trainable = False

        encoder, momentum_encoder = self._construct_heads(
            fastfcn,
            momentum_fastfcn,
            parameterization
        )

        model = rs.models.fastfcn_moco_context.FastFCNMoCoContextTrainingModel(
            encoder=encoder,
            momentum_encoder=momentum_encoder,
            momentum=parameterization['moco_momentum'],
            temperature=parameterization['moco_initial_temperature'],
            queue_size=parameterization['moco_queue_size'],
            features=parameterization['moco_mlp_features']
        )

        model.build(list(map(lambda spec: spec.shape, training_dataset.element_spec[0])))

        steps_per_epoch = np.ceil(len(supervised_training_masks) / parameterization['segmentation_batch_size'])
        optimizer = self.keras.build_optimizer(
            total_steps=parameterization['max_epochs'] * steps_per_epoch,
            initial_learning_rate=np.power(10.0, parameterization['initial_learning_rate_exp']),
            end_learning_rate=np.power(10.0, parameterization['end_learning_rate_exp']),
            learning_rate_decay=parameterization['learning_rate_decay'],
            momentum=parameterization['momentum'],
            weight_decay=parameterization['weight_decay']
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
            'output_1': parameterization['segmentation_loss_ratio'],
            'output_2': 1.0 - parameterization['segmentation_loss_ratio']
        }

        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

        callbacks = [
            self.keras.decay_temperature_callback(
                initial_temperature=parameterization['moco_initial_temperature'],
                min_temperature=parameterization['moco_min_temperature'],
                decay_rate=parameterization['moco_temperature_decay']
            )
        ] + model.create_callbacks()  # For MoCo updates

        model.fit(
            training_dataset,
            epochs=parameterization['max_epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=2 if self.parameters['base_is_debug'] else 0  # One line/epoch if debug, no output otherwise
        )

        # Evaluate model
        validation_scores = []
        for validation_image, (validation_mask, _) in validation_dataset_large:
            raw_predicted_mask, _ = model.predict(validation_image)
            predicted_mask = np.where(raw_predicted_mask >= 0, 1., 0.)
            predicted_mask = tf.round(
                rs.data.cil.segmentation_to_patch_labels(
                    predicted_mask,
                    model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE
                )[0].astype(np.int)
            )
            predicted_mask = predicted_mask.numpy().astype(int)
            validation_mask = rs.data.cil.segmentation_to_patch_labels(validation_mask.numpy()).astype(np.int)
            validation_scores.append(
                sklearn.metrics.accuracy_score(validation_mask.flatten(), predicted_mask.flatten())
            )

        return float(np.mean(validation_scores))

    def _load_datasets(
            self,
            training_images: np.ndarray,
            training_masks: np.ndarray,
            validation_images: np.ndarray,
            validation_masks: np.ndarray,
            unsupervised_training_sample_paths: np.ndarray
    ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        # Uses all unsupervised samples in a flat order
        unlabelled_dataset = rs.data.unsupervised.shuffled_image_dataset(
            unsupervised_training_sample_paths,
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
                               rs.data.unsupervised.augment_patch(
                                   image,
                                   crop_size=self.parameters['moco_augmentation_crop_size']
                               ),
                               rs.data.unsupervised.augment_patch(
                                   image,
                                   crop_size=self.parameters['moco_augmentation_crop_size']
                               )
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

        # Validation images can be directly converted to the model colour space
        validation_dataset_large = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(validation_images), validation_masks)
        )

        zeros_image = tf.zeros((1,) + self.parameters['moco_augmentation_crop_size'], dtype=tf.float32)
        zeros_dataset = tf.data.Dataset.from_tensor_slices((zeros_image, [0]))
        zeros_dataset = zeros_dataset.repeat(len(validation_images))
        validation_dataset_large = tf.data.Dataset.zip((zeros_dataset, validation_dataset_large))

        # Tuples are structured as follows:
        # zeros: (zero placeholder image, zero placeholder label)
        # labelled: (labelled image, segmentation mask)
        # Since evaluation the contrastive loss is not useful, we use dummy inputs.
        # Result is a single evaluation batch:
        # ((labelled input image, zero placeholder image, zero placeholder image),
        #   (segmentation mask label, contrastive loss label)
        # )
        validation_dataset = validation_dataset_large.map(
            lambda zeros, labelled: (
                (labelled[0], zeros[0], zeros[0]),
                (rs.data.cil.resize_mask_to_stride(labelled[1], rs.models.fastfcn.OUTPUT_STRIDE), zeros[1])
            )
        )
        validation_dataset_large = validation_dataset_large.map(
            lambda zeros, labelled: (
                (labelled[0], zeros[0], zeros[0]),
                (labelled[1], zeros[1])
            )
        )
        validation_dataset_large = validation_dataset_large.batch(1)
        validation_dataset = validation_dataset.batch(1)

        return training_dataset, validation_dataset, validation_dataset_large

    def _construct_fastfcn(self, parameterization: dict) -> tf.keras.Model:
        # The original does shuffling batch norm across GPUs to avoid issues stemming from
        #  leaking statistics via the normalization.
        #  We need a solution which works on a single GPU.
        #  arXiv:1905.09272 [cs.CV] does layer norm instead which is more suitable to our single-GPU case.
        #  This seems to fit similarly fast but we need to evaluate the effects on downstream performance.
        #  Furthermore, layer normalization requires much more memory than batch norm and thus reduces batch size etc.
        normalization_builder = rs.util.LayerNormalizationBuilder()

        resnet_kwargs = {
            'kernel_initializer': parameterization['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

        name = parameterization['backbone']
        backbone = None
        if name == 'ResNet50':
            backbone = rs.models.resnet.ResNet50Backbone(**resnet_kwargs)
        if name == 'ResNet101':
            backbone = rs.models.resnet.ResNet101Backbone(**resnet_kwargs)

        if backbone is None:
            raise AssertionError(f'Unexpected backbone name "{name}"')

        fastfcn = rs.models.fastfcn.FastFCN(
            backbone,
            jpu_features=parameterization['jpu_features'],
            se_loss_features=parameterization['se_loss_features'],
            head_dropout_rate=parameterization['head_dropout'],
            dense_initializer=parameterization['dense_initializer'],
            kernel_initializer=parameterization['kernel_initializer'],
        )
        return fastfcn

    def _construct_heads(
            self,
            fastfcn: tf.keras.Model,
            momentum_fastfcn: tf.keras.Model,
            parameterization: dict
    ) -> typing.Tuple[tf.keras.layers.Layer, tf.keras.layers.Layer]:
        # MoCo v2 head
        encoder = rs.models.fastfcn_moco_context.FastFCNMoCoContextMLPHead(
            fastfcn,
            output_features=parameterization['moco_mlp_features'],
            dense_initializer=parameterization['dense_initializer'],
            name='encoder'
        )
        momentum_encoder = rs.models.fastfcn_moco_context.FastFCNMoCoContextMLPHead(
            momentum_fastfcn,
            output_features=parameterization['moco_mlp_features'],
            name='momentum_encoder',
            trainable=False
        )

        return encoder, momentum_encoder


if __name__ == '__main__':
    main()
