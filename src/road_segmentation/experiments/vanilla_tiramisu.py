import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Vanilla Tiramisu Baseline'
EXPERIMENT_TAG = 'baseline_vanilla_tiramisu'


class VanillaTiramisu(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Not passing any parameters will run the FC-DenseNet103 with the same parameters as in the papers,
        # and is one of our baseline experiments.
        parser.add_argument('--batch-size', type=int, default=3, help='Training batch size.')
        parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate.')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')
        parser.add_argument('--learning-rate-finetune', type=float, default=1e-4,
                            help='Learning rate for the restarted training session.')
        parser.add_argument('--exponential-decay', type=float, default=0.995,
                            help='Exponential decay for learning rate after each epoch.')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Factor for weight decay regularisers.')
        parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs.')
        parser.add_argument('--patience-metric', type=str, default='val_binary_mean_iou_score',
                            choices=['binary_mean_iou_score', 'binary_mean_accuracy', 'binary_mean_f_score',
                                     'val_loss', 'val_binary_mean_accuracy', 'val_binary_mean_f_score',
                                     'val_binary_mean_iou_score', 'loss'],
                            help='Metric to be used by patience early stopping.')
        parser.add_argument('--fcnet', type=int, default=103, choices=[56, 67, 103],
                            help='Specify model to be used. 0: tiny testing net, 56/67/103: FC-DenseNet56/67/103 respectively.')
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
            'patience_metric': args.patience_metric,
            'fcnet': args.fcnet
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
        self.log.info('Loading training and validation data')

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
            return

        finetune_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))

        def tiramisu_augmentations(image, mask):
            """Applies some basic data augmentation as described in the paper.

            First randomly flips the image and mask horizontally, then vertically and then randomly crops the image and
            mask to size 192x192. This is size is chosen such that, during training, the model will never have to crop
            outputs of upsampling layers.

            Args:
                image: The training image.
                mask: The training mask.
            """
            target_width = 192
            target_height = 192
            flip_left_right = np.random.choice([True, False])
            flip_up_down = np.random.choice([True, False])
            x_offset = np.random.randint(0, image.shape[0] - target_width)
            y_offset = np.random.randint(0, image.shape[1] - target_height)
            if flip_left_right:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)

            if flip_up_down:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)

            image = tf.image.crop_to_bounding_box(
                image,
                x_offset,
                y_offset,
                target_height,
                target_width
            )
            mask = tf.image.crop_to_bounding_box(
                mask,
                x_offset,
                y_offset,
                target_height,
                target_width
            )
            return image, mask

        # First normal training is done, then fine tuning. Finetuning is done on the standard data set while training
        # is done on the data set augmented with random crops and flips on both axes. The crops essentially reduce the
        # training data set by a factor of a bit more than 4, as the cropped data are a bit less than a fourth the size
        # of the original training data. To compensate for that, first 4 copies of the data set are concatenated into
        # the training data set, and then the augmentations are applied. This should result in a training data set
        # of approximately equal size (and 4 times the sample count).

        # first doubling of size.
        training_dataset = finetune_dataset.concatenate(finetune_dataset)

        # second doubling of size -> 4 times original data set.
        training_dataset = training_dataset.concatenate(training_dataset)

        # apply augmentations.
        training_dataset = training_dataset.map(tiramisu_augmentations)

        # shuffle and batch the datasets before use.
        finetune_dataset = finetune_dataset.shuffle(buffer_size=1024)
        finetune_dataset = finetune_dataset.batch(batch_size)

        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(batch_size)

        self.log.debug('Training data specification: %s', training_dataset.element_spec)
        self.log.debug('Finetune data specification: %s', finetune_dataset.element_spec)

        # Load validation data
        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        model = rs.models.tiramisu.build_fc_dense_net(
            model=self.parameters['fcnet'],
            dropout_rate=self.parameters['dropout_rate'],
            weight_decay=self.parameters['weight_decay']
        )

        lr = self.parameters['learning_rate']

        def exp_epoch_decay_sched(epoch):
            """Exponentially lowers the learning rate by some factor.
            """
            de = self.parameters['exponential_decay']
            lr_new = lr * tf.pow(de, epoch)
            self.log.debug("epoch: %d, lr: %f, de: %f: lr_new: %f", epoch, lr, de, lr_new)
            return lr_new

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

        if self.parameters['patience_metric'] in ['val_loss', 'loss']:
            patience_mode = 'min'
        else:
            patience_mode = 'max'

        self.log.info("Starting training")

        hist = model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks +
                      # Learning rate decay is only used in the first training and the patience parameters change for
                      # finetuning, hence we append these callbacks here to the default callbacks.
                      [
                          tf.keras.callbacks.EarlyStopping(
                              monitor=self.parameters['patience_metric'],
                              min_delta=0,
                              patience=self.parameters['patience'],
                              mode=patience_mode
                          ),
                          tf.keras.callbacks.LearningRateScheduler(exp_epoch_decay_sched)
                      ]
        )

        self.log.info("Loading best model")

        # Load best weights and find epoch in which training previously stopped.
        model.load_weights(self.keras.default_best_checkpoint_path())

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.parameters['learning_rate_finetune']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.keras.default_metrics(threshold=0.0),
        )

        self.log.info("Starting fine tuning")

        model.fit(
            finetune_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks +
                      # Explicitly append an early stopping callback with the patience for fine tuning parameters.
                      [
                          tf.keras.callbacks.EarlyStopping(
                              monitor=self.parameters['patience_metric'],
                              min_delta=0,
                              patience=self.parameters['patience_finetune'],
                              mode=patience_mode
                          )
                      ],
            initial_epoch=len(hist.epoch)
        )

        self.log.info("Training done. Loading best model.")

        # Load the best model, which is the one that is considered
        # the output of the vanilla tiramisu training procedure.
        model.load_weights(self.keras.default_best_checkpoint_path())

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()
        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)

            target_height = image.shape[0] // rs.data.cil.PATCH_SIZE
            target_width = image.shape[1] // rs.data.cil.PATCH_SIZE

            image = np.asanyarray([image])
            predictions = classifier.predict(image)
            prediction_mask = tf.sigmoid(predictions)
            prediction_mask_patches = rs.data.cil.segmentation_to_patch_labels(prediction_mask)
            result[sample_id] = prediction_mask_patches[0]

            assert result[sample_id].shape == (target_height, target_width)

        return result


def main():
    VanillaTiramisu().run()


if __name__ == '__main__':
    main()
