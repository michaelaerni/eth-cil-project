import argparse
import os
import typing

import ax
import ax.modelbridge.generation_strategy
import numpy as np
import sklearn.model_selection
import tensorflow as tf
import tensorflow_addons as tfa

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN Baseline Parameter Search'
EXPERIMENT_TAG = 'baseline_fastfcn_search'


def main():
    BaselineFCNSearchExperiment().run()


class BaselineFCNSearchExperiment(rs.framework.Experiment):

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
        parser.add_argument('--initial-trials', type=int, default=8, help='Number of initial SOBOL trials')
        parser.add_argument('--optimised-trials', type=int, default=12, help='Number of GP trials')
        parser.add_argument('--folds', type=int, default=5, help='Number of CV folds per trial')
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
            'batch_size': args.batch_size,
            'max_epochs': args.epochs,
            'initial_trials': args.initial_trials,
            'optimised_trials': args.optimised_trials,
            'folds': args.folds,
            'prefetch_buffer_size': 16,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'augmentation_interpolation': 'bilinear',
            'augmentation_blur_probability': 0.5,
            'augmentation_blur_size': 5,  # 5x5 Gaussian filter for blurring
            'training_image_size': (384, 384)
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training and validation data')
        try:
            trainig_paths, validation_paths = rs.data.cil.train_validation_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(trainig_paths)
            validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            self.log.debug(
                'Loaded %d training and %d validation samples',
                training_images.shape[0],
                validation_images.shape[0]
            )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        self.log.info('Building experiment')
        search_space = self._build_search_space()
        self.log.debug('Built search space %s', search_space)

        # TODO: Might change this to accuracy following Kaggle
        objective = ax.Objective(metric=ax.Metric('f1_score', lower_is_better=False), minimize=False)
        optimization_config = ax.OptimizationConfig(objective, outcome_constraints=None)
        self.log.debug('Built optimization config %s', optimization_config)

        def _evaluation_function_wrapper(
                parameterization: typing.Dict[str, typing.Union[float, str, bool, int]],
                weight: typing.Optional[float] = None
        ) -> typing.Dict[str, typing.Tuple[float, float]]:
            return self._run_trial(parameterization, training_images, training_masks)

        # TODO: How should we handle model validation?
        #  Possibility 1: Use cross-validation on training data, don't use validation set
        #  Possibility 2: Use cross-validation on training and validation data
        #  Possibility 3: Only train once on training set, predict on validation set

        # TODO: Include dry-run option
        experiment = ax.SimpleExperiment(
            search_space=search_space,
            name=self.tag,
            evaluation_function=_evaluation_function_wrapper
        )
        experiment.optimization_config = optimization_config
        self.log.debug('Built experiment %s', experiment)

        generation_strategy = self._build_generation_strategy()
        self.log.info('Using generation strategy %s', generation_strategy)

        loop = ax.OptimizationLoop(
            experiment,
            total_trials=self.parameters['initial_trials'] + self.parameters['optimised_trials'],
            arms_per_trial=1,
            random_seed=self.SEED,
            wait_time=0,
            run_async=False,
            generation_strategy=generation_strategy
        )
        self.log.info('Running trials')
        loop.full_run()
        self.log.info('Finished all trials')

        best_parameterization, (means, covariances) = loop.get_best_point()
        self.log.info('Best encountered parameters: %s', best_parameterization)
        self.log.info('Best encountered score: mean=%.4f, var=%.4f', means['f1_score'], covariances['f1_score']['f1_score'])

        experiment_save_path = os.path.join(self.experiment_directory, 'trials.json')
        ax.save(experiment, experiment_save_path)
        self.log.info('Saved experiment to %s', experiment_save_path)

        # TODO: Some way to plot the results (optimisation function), either here or in a notebook

        # TODO: Finish experiment
        raise NotImplementedError()

        return model

    def _build_search_space(self) -> ax.SearchSpace:
        # Inclusive lower bound for log-scale parameters with lower bound 0
        epsilon = np.finfo(np.float).eps

        # Having fixed parameters here should not have any negative impact

        # TODO: Build full search space
        # TODO: Include discrete parameters
        # TODO: Many parameters are fixed even though they could be searched over
        initial_learning_rate_exp = ax.RangeParameter('initial_learning_rate_exp', ax.ParameterType.FLOAT, lower=-5.0, upper=0.0)
        end_learning_rate_exp = ax.FixedParameter('end_learning_rate_exp', ax.ParameterType.FLOAT, value=-8.0)
        parameters = [
            ax.FixedParameter('jpu_features', ax.ParameterType.INT, value=512),
            ax.FixedParameter('max_epochs', ax.ParameterType.INT, value=self.parameters['max_epochs']),
            ax.FixedParameter('batch_size', ax.ParameterType.INT, value=self.parameters['batch_size']),
            ax.FixedParameter('backbone', ax.ParameterType.STRING, value=self.parameters['backbone']),
            ax.FixedParameter('weight_decay', ax.ParameterType.FLOAT, value=1e-4),
            ax.RangeParameter('head_dropout', ax.ParameterType.FLOAT, lower=0.0, upper=0.5),
            ax.RangeParameter('segmentation_loss_ratio', ax.ParameterType.FLOAT, lower=0.0, upper=1.0),
            ax.FixedParameter('output_upsampling', ax.ParameterType.STRING, value='nearest'),
            ax.FixedParameter('kernel_initializer', ax.ParameterType.STRING, value='he_normal'),  # FIXME: This might not necessarily be the best choice
            ax.FixedParameter('dense_initializer', ax.ParameterType.STRING, value='he_uniform'),  # Only for the dense weights in the Encoder head
            initial_learning_rate_exp,
            end_learning_rate_exp,
            ax.FixedParameter('learning_rate_decay', ax.ParameterType.FLOAT, value=0.9),
            ax.FixedParameter('momentum', ax.ParameterType.FLOAT, value=0.9)
        ]

        parameter_constraints = []

        return ax.SearchSpace(parameters, parameter_constraints)

    def _build_generation_strategy(self) -> ax.modelbridge.generation_strategy.GenerationStrategy:
        return ax.modelbridge.generation_strategy.GenerationStrategy([
            ax.modelbridge.generation_strategy.GenerationStep(
                model=ax.Models.SOBOL,
                num_trials=self.parameters['initial_trials'],
                enforce_num_trials=True,
                model_kwargs={
                    'deduplicate': True,
                    'seed': self.SEED
                }
            ),
            ax.modelbridge.generation_strategy.GenerationStep(
                model=ax.Models.GPEI,
                num_trials=self.parameters['optimised_trials'],
                enforce_num_trials=True
            )
        ])

    def _run_trial(
            self,
            parameterization: typing.Dict[str, typing.Union[float, str, bool, int]],
            images: np.ndarray,
            masks: np.ndarray
    ) -> typing.Dict[str, typing.Tuple[float, float]]:
        self.log.debug('Current trial parameters: %s', parameterization)

        # TODO: Store logs somewhere (for long training sessions)

        # TODO: Make metric customizable or use different one?

        # TODO: How to do early stopping? And implement it.

        fold_generator = sklearn.model_selection.KFold(
            n_splits=self.parameters['folds'],
            shuffle=True,
            random_state=self.SEED
        )
        fold_scores = []
        for fold_idx, (training_indices, validation_indices) in enumerate(fold_generator.split(images)):
            self.log.info('Running fold %d / %d', fold_idx + 1, self.parameters['folds'])

            current_score = self._run_fold(
                parameterization,
                images[training_indices],
                masks[training_indices],
                images[validation_indices],
                masks[validation_indices]
            )
            fold_scores.append(current_score)
            self.log.debug('Fold completed, score: %.4f', current_score)

        score_mean = np.mean(fold_scores)
        score_std = np.std(fold_scores)
        score_sem = score_std / np.sqrt(self.parameters['folds'])
        self.log.info('Finished trial with score %.4f (std %.4f, sem %.4f)', score_mean, score_std, score_sem)

        return {
            'f1_score': (float(score_mean), float(score_sem))
        }

    def _run_fold(
            self,
            parameterization: typing.Dict[str, typing.Union[float, str, bool, int]],
            training_images: np.ndarray,
            training_masks: np.ndarray,
            validation_images: np.ndarray,
            validation_masks: np.ndarray
    ) -> float:
        # TODO: Ensure each fold receives the correct data and the data set repeats correctly
        # TODO: At this point we should move data preprocessing into a central place, at least per model
        #  but ideally as global as possible. That way, each model receives more or less the same data.
        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        training_dataset = training_dataset.map(lambda image, mask: self._augment_sample(image, mask))
        training_dataset = training_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        training_dataset = training_dataset.batch(parameterization['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])

        # Validation images can be directly converted to the model colour space
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(validation_images), validation_masks)
        )
        validation_dataset = validation_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        backbone = self._construct_backbone(parameterization['backbone'], parameterization['kernel_initializer'])
        model = rs.models.fastfcn.FastFCN(
            backbone,
            parameterization['jpu_features'],
            parameterization['head_dropout'],
            parameterization['kernel_initializer'],
            parameterization['dense_initializer'],
            parameterization['output_upsampling'],
            kernel_regularizer=None
        )
        model.build(training_dataset.element_spec[0].shape)

        metrics = {
            'output_1': self.keras.default_metrics(threshold=0.0)
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

        steps_per_epoch = np.ceil(training_images.shape[0] / parameterization['batch_size'])
        optimizer = self._build_optimizer(steps_per_epoch, parameterization)

        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        callbacks = [
            # TODO: tf.keras.callbacks.EarlyStopping()
        ]

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['max_epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks,
            verbose=2 if self.parameters['base_is_debug'] else 0  # One line/epoch if debug, no output otherwise
        )

        # Evaluate model
        validation_scores = []
        for validation_image, (validation_mask, _) in validation_dataset:
            raw_predicted_mask, _ = model.predict(validation_image)
            predicted_mask = np.where(raw_predicted_mask >= 0.0, 1, 0)
            predicted_mask = rs.data.cil.segmentation_to_patch_labels(predicted_mask)[0].astype(np.int)
            validation_mask = rs.data.cil.segmentation_to_patch_labels(validation_mask.numpy())[0].astype(np.int)
            validation_scores.append(sklearn.metrics.f1_score(validation_mask.flatten(), predicted_mask.flatten()))

        # FIXME: It would be better to compare the mean of deltas instead of the delta of means! cf: JMB
        return float(np.mean(validation_scores))

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            # Convert to model colour space
            image = rs.data.image.rgb_to_cielab(image)

            (raw_prediction,), _ = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)

            result[sample_id] = prediction

        return result

    def _augment_sample(self, image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        # Random Gaussian blurring
        do_blur = tf.random.uniform(shape=[], dtype=tf.float32) < self.parameters['augmentation_blur_probability']
        blurred_image = tf.cond(do_blur, lambda: self._augment_blur(image), lambda: image)
        blurred_image.set_shape(image.shape)  # Must set shape manually since it cannot be inferred from tf.cond

        # Random scaling
        scaling_factor = tf.random.uniform(
            shape=[],
            minval=1.0 - self.parameters['augmentation_max_relative_scaling'],
            maxval=1.0 + self.parameters['augmentation_max_relative_scaling']
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
        crop_size = self.parameters['training_image_size'] + (4,)  # 3 colour channels + 1 mask channel
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
        kernel_size = self.parameters['augmentation_blur_size']
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

    def _build_optimizer(
            self,
            steps_per_epoch: int,
            parameterization: typing.Dict[str, typing.Union[float, str, bool, int]]
    ) -> tfa.optimizers.SGDW:
        # TODO: This performs weight decay on an optimizer level, not on a case-by-case basis.
        #  There's a difference!
        #  Global weight decay might be dangerous if we also have the Encoder head (with the parameters there)
        #  but it could also be an important ingredient for success...

        initial_learning_rate = np.power(10.0, parameterization['initial_learning_rate_exp'])
        end_learning_rate = np.power(10.0, parameterization['end_learning_rate_exp'])

        learning_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=self.parameters['max_epochs'] * steps_per_epoch,
            end_learning_rate=end_learning_rate,
            power=parameterization['learning_rate_decay']
        )

        # Determine the weight decay schedule proportional to the learning rate decay schedule
        weight_decay_factor = parameterization['weight_decay'] / initial_learning_rate

        # This has to be done that way since weight_decay needs to access the optimizer lazily, hence the lambda
        optimizer = tfa.optimizers.SGDW(
            weight_decay=lambda: weight_decay_factor * learning_rate_scheduler(optimizer.iterations),
            learning_rate=learning_rate_scheduler,
            momentum=parameterization['momentum']
        )

        return optimizer


if __name__ == '__main__':
    main()
