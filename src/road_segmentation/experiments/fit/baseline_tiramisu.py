import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Tiramisu Baseline'
EXPERIMENT_TAG = 'baseline_tiramisu'

TRAINING_TARGET_DIMENSION = 192
"""
Images and masks are cropped to this width and height for training.
"""

EARLY_STOP_METRIC = 'val_binary_mean_accuracy'
"""
Metric to be used for early stopping training and finetuning.
"""


def exp_epoch_decay_sched(exponential_decay: float, learning_rate: float) -> typing.Callable[[int], float]:
    """
    Returns a lambda function which exponentially lowers the learning rate by some factor, to be used with the
    Keras LearningRateScheduler callback.

    Args:
        exponential_decay: The decay rate.
        learning_rate: The learning rate which the training starts with.

    Returns:
        A lambda which can be passed to the Keras LearningRateScheduler callback.
    """
    return lambda epoch: learning_rate * tf.pow(exponential_decay, epoch)


class TiramisuExperiment(rs.framework.FitExperiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Not passing any parameters will run the FC-DenseNet103 with the same parameters as in the papers,
        # and is one of our baseline experiments.
        parser.add_argument(
            '--batch-size',
            type=int,
            default=2,  # Original was 3 but we can only afford 2
            help='Training batch size.'
        )
        parser.add_argument(
            '--dropout-rate',
            type=float,
            default=0.2,
            help='Dropout rate.'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-3,
            help='Learning rate.'
        )
        parser.add_argument(
            '--learning-rate-finetune',
            type=float,
            default=1e-4,
            help='Learning rate for the restarted training session.'
        )
        parser.add_argument(
            '--exponential-decay',
            type=float,
            default=0.995,
            help='Exponential decay for learning rate after each epoch.'
        )
        parser.add_argument(
            '--weight-decay',
            type=float,
            default=1e-4,
            help='Factor for weight decay regularisers.'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=300,
            help='Number of training epochs.'
        )
        parser.add_argument(
            '--model-type',
            type=str,
            default="FCDenseNet103",
            choices=['FCDenseNet103', 'FCDenseNet67', 'FCDenseNet56'],
            help='Specify the model to be used. Choices are: FCDenseNet56, FCDenseNet67 or FCDenseNet103.')
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'learning_rate_finetune': args.learning_rate_finetune,
            'exponential_decay': args.exponential_decay,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'patience': 100,
            'patience_finetune': 50,
            'model_type': args.model_type
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
        self.log.info('Loading training and validation data')

        training_images, training_masks, validation_images, validation_masks = self._load_data_images(
            self.data_directory
        )

        training_dataset, finetune_dataset, validation_dataset = self._build_data_sets(
            batch_size,
            training_images,
            training_masks,
            validation_images,
            validation_masks
        )

        self.log.debug('Training data specification: %s', training_dataset.element_spec)
        self.log.debug('Finetune data specification: %s', finetune_dataset.element_spec)
        self.log.debug('Validation data specification: %s', validation_dataset.element_spec)

        model = self._build_model()

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.log_predictions(validation_images),
            self.keras.best_checkpoint_callback(),
        ]

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.parameters['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.keras.default_metrics(threshold=0.0)
        )

        self.log.info("Starting training")

        # Learning rate decay is only used in the first training and the patience parameters change for
        # finetuning, hence we append these callbacks here to the default callbacks.
        training_callbacks = callbacks + [
            tf.keras.callbacks.EarlyStopping(
                monitor=EARLY_STOP_METRIC,
                min_delta=0,
                patience=self.parameters['patience'],
                mode='max'
            ),
            tf.keras.callbacks.LearningRateScheduler(
                exp_epoch_decay_sched(
                    self.parameters['exponential_decay'],
                    self.parameters['learning_rate']
                )
            )
        ]

        # Explicitly append an early stopping callback with the patience for fine tuning parameters.
        finetune_callbacks = callbacks + [
            tf.keras.callbacks.EarlyStopping(
                monitor=EARLY_STOP_METRIC,
                min_delta=0,
                patience=self.parameters['patience_finetune'],
                mode='max'
            )
        ]

        training_history = model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=training_callbacks
        )

        self.log.info("Loading best model")

        # Load best weights found during training.
        model.load_weights(self.keras.default_best_checkpoint_path())

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.parameters['learning_rate_finetune']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.keras.default_metrics(threshold=0.0),
        )

        self.log.info("Starting fine tuning")

        model.fit(
            finetune_dataset,
            epochs=self.parameters['epochs'] + len(training_history.epoch),
            validation_data=validation_dataset,
            callbacks=finetune_callbacks,
            initial_epoch=len(training_history.epoch)
        )

        self.log.info("Training done. Loading best model.")

        # Load best weights found during finetuning. This is considered the output of the tiramisu experiment.
        model.load_weights(self.keras.default_best_checkpoint_path())

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()
        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)

            image = np.expand_dims(image, axis=0)

            raw_prediction = classifier.predict(image)

            # predict labels from logits
            prediction= np.where(raw_prediction >= 0, 1.0, 0.0)

            # Threshold patches to create final prediction
            prediction = rs.data.cil.segmentation_to_patch_labels(prediction)[0]
            prediction = prediction.astype(int)

            result[sample_id] = prediction.astype(int)

        return result

    def _build_model(self):
        """
        Builds the model based on parameter.
        Returns:
            The built model
        """
        self.log.info('Building model')

        if self.parameters['model_type'] == 'FCDenseNet56':
            model = rs.models.tiramisu.TiramisuFCDenseNet56(
                dropout_rate=self.parameters['dropout_rate'],
                weight_decay=self.parameters['weight_decay']
            )
        elif self.parameters['model_type'] == 'FCDenseNet67':
            model = rs.models.tiramisu.TiramisuFCDenseNet67(
                dropout_rate=self.parameters['dropout_rate'],
                weight_decay=self.parameters['weight_decay']
            )
        elif self.parameters['model_type'] == 'FCDenseNet103':
            model = rs.models.tiramisu.TiramisuFCDenseNet103(
                dropout_rate=self.parameters['dropout_rate'],
                weight_decay=self.parameters['weight_decay']
            )
        else:
            raise AssertionError(
                "Unexpected model type self.parameters['model_type'] = " + self.parameters['model_type']
            )
        return model

    @staticmethod
    def _load_data_images(
            data_directory: str
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load images from disk into numpy arrays.
        Args:
            data_directory: The directory where the image data is located.

        Returns:
            A 4 tuple, of training images and masks as well as validation images and masks
        """
        training_paths, validation_paths = rs.data.cil.train_validation_sample_paths(data_directory)
        training_images, training_masks = rs.data.cil.load_images(training_paths)
        validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
        return training_images, training_masks, validation_images, validation_masks

    @staticmethod
    def _build_data_sets(
            batch_size: int,
            training_images: np.ndarray,
            training_masks: np.ndarray,
            validation_images: np.ndarray,
            validation_masks: np.ndarray
    ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Builds TensorFlow data sets for from raw data.
        Args:
            batch_size: Batch size to be used in training.
            training_images: RGB Training images.
            training_masks: Black and white training masks.
            validation_images: RGB validation images.
            validation_masks: Black and white validation masks.

        Returns:
            3-tuple of datasets: training dataset, finetune dataset and validation dataset.
        """
        finetune_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = finetune_dataset.map(tiramisu_augmentations)

        finetune_dataset = finetune_dataset.shuffle(buffer_size=training_images.shape[0])
        finetune_dataset = finetune_dataset.batch(batch_size)

        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        training_dataset = training_dataset.batch(batch_size)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        return training_dataset, finetune_dataset, validation_dataset


def tiramisu_augmentations(image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies some basic data augmentation as described in the paper.

    First randomly flips the image and mask horizontally, then vertically and then randomly crops the image and
    mask to size 192x192. This is size is chosen such that, during training, the model will never have to crop
    outputs of upsampling layers.

    Args:
        image: The training image.
        mask: The training mask.
    Returns:
        A tuple of the augmented image and mask.
    """
    to_flip = tf.concat([image, mask], -1)
    to_flip = tf.image.random_flip_left_right(to_flip)
    flipped = tf.image.random_flip_up_down(to_flip)
    cropped = tf.image.random_crop(flipped, [TRAINING_TARGET_DIMENSION, TRAINING_TARGET_DIMENSION, 4])
    image = cropped[:, :, :3]
    mask = cropped[:, :, -1:]

    return image, mask


def main():
    TiramisuExperiment().run()


if __name__ == '__main__':
    main()
