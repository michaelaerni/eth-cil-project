import argparse
import typing

import ax
import numpy as np
import sklearn.model_selection
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN Baseline Parameter Search'
EXPERIMENT_TAG = 'baseline_fastfcn_search'


def main():
    BaselineFastFCNSearchExperiment().run()


class BaselineFastFCNSearchExperiment(rs.framework.SearchExperiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on ADE20k experiments of the FastFCN paper
        parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')  # FIXME: Was 16 originally
        parser.add_argument('--epochs', type=int, default=120, help='Maximum number of training epochs per fold')
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )
        parser.add_argument('--prefetch-buffer-size', type=int, default=16, help='Number of batches to pre-fetch')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'backbone': args.backbone,
            'batch_size': args.batch_size,
            'max_epochs': args.epochs,
            'prefetch_buffer_size': args.prefetch_buffer_size,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'training_image_size': (384, 384)
        }

    def build_search_space(self) -> ax.SearchSpace:
        # FIXME: Many parameters are fixed even though they could be searched over

        parameters = [
            ax.FixedParameter('jpu_features', ax.ParameterType.INT, value=512),
            ax.FixedParameter('max_epochs', ax.ParameterType.INT, value=self.parameters['max_epochs']),
            ax.FixedParameter('batch_size', ax.ParameterType.INT, value=self.parameters['batch_size']),
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
            ax.FixedParameter('momentum', ax.ParameterType.FLOAT, value=0.9)
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
        training_dataset = tf.data.Dataset.from_tensor_slices((supervised_training_images, supervised_training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=supervised_training_images.shape[0])
        training_dataset = training_dataset.map(lambda image, mask: rs.data.cil.augment_image(
            image,
            mask,
            crop_size=self.parameters['training_image_size'],
            max_relative_scaling=self.parameters['augmentation_max_relative_scaling'],
            model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE
        ))
        training_dataset = training_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        training_dataset = training_dataset.batch(parameterization['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])

        # Validation images can be directly converted to the model colour space
        # We need to keep the larger dataset, so that we can later evaluate the model with more accurate
        # downsampling + thresholding.
        validation_dataset_large = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(supervised_validation_images), supervised_validation_masks)
        )
        validation_dataset = validation_dataset_large.map(
            lambda image, mask: (image, rs.data.cil.resize_mask_to_stride(mask, rs.models.fastfcn.OUTPUT_STRIDE))
        )
        validation_dataset_large = validation_dataset_large.map(
            lambda image, mask: self._calculate_se_loss_target(image, mask)
        )
        validation_dataset = validation_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        validation_dataset_large = validation_dataset_large.batch(1)
        validation_dataset = validation_dataset.batch(1)

        # Build model
        backbone = self._construct_backbone(parameterization['backbone'], parameterization['kernel_initializer'])
        model = rs.models.fastfcn.FastFCN(
            backbone,
            parameterization['jpu_features'],
            parameterization['head_dropout'],
            parameterization['kernel_initializer'],
            parameterization['dense_initializer'],
            kernel_regularizer=None
        )
        model.build(training_dataset.element_spec[0].shape)

        metrics = {
            'output_1': self.keras.default_metrics(threshold=0.0, model_output_stride=rs.models.fastfcn.OUTPUT_STRIDE)
        }

        # TODO: Check whether the binary cross-entropy loss behaves correctly
        losses = {
            'output_1': tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Segmentation loss
            'output_2': tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Modified SE-loss
        }
        loss_weights = {
            'output_1': parameterization['segmentation_loss_ratio'],
            'output_2': 1.0 - parameterization['segmentation_loss_ratio']
        }

        steps_per_epoch = np.ceil(supervised_training_images.shape[0] / parameterization['batch_size'])
        optimizer = self.keras.build_optimizer(
            total_steps=self.parameters['max_epochs'] * steps_per_epoch,
            initial_learning_rate=np.power(10.0, parameterization['initial_learning_rate_exp']),
            end_learning_rate=np.power(10.0, parameterization['end_learning_rate_exp']),
            learning_rate_decay=parameterization['learning_rate_decay'],
            momentum=parameterization['momentum'],
            weight_decay=parameterization['weight_decay']
        )

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['max_epochs'],
            validation_data=validation_dataset,
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

    # noinspection PyMethodMayBeStatic
    def _calculate_se_loss_target(
            self,
            image: tf.Tensor,
            mask: tf.Tensor
    ) -> typing.Tuple[tf.Tensor, typing.Tuple[tf.Tensor, tf.Tensor]]:
        # The target to predict is the logit of the proportion of foreground pixels (i.e. empirical prior)
        foreground_prior = tf.reduce_mean(mask)

        return image, (mask, foreground_prior)

    @classmethod
    def _construct_backbone(cls, name: str, kernel_initializer: str) -> tf.keras.Model:
        if name == 'ResNet50':
            return rs.models.resnet.ResNet50Backbone(
                kernel_initializer=kernel_initializer
            )
        if name == 'ResNet101':
            return rs.models.resnet.ResNet101Backbone(
                kernel_initializer=kernel_initializer
            )

        raise AssertionError(f'Unexpected backbone name "{name}"')


if __name__ == '__main__':
    main()
