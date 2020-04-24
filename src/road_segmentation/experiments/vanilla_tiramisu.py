import argparse
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Vanilla Tiramisu'
EXPERIMENT_TAG = 'vanilla_tiramisu'


class VanillaTiramisu(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--exponential-decay', type=float, default=0.995,
                            help='Exponential decay for learning rate after each epoch')
        # parser.add_argument('--fcnet', type=int, default=103, choices=[0, 56, 67, 102], help='')
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'exponential_decay': args.exponential_decay,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
        self.log.info('Loading training and validation data')

        # TODO: - weight decay of 1e-4,
        #       - Monitoring metrics with patience of 100
        #       - Early stopping
        #       - Fine tuning with learning rate 1e-4 and patience of 50

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

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(batch_size)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')

        model = rs.models.tiramisu.build_FCDenseNet103(dropout_rate=self.parameters['dropout_rate'])

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.parameters['learning_rate']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.keras.default_metrics(threshold=0.0)
        )

        # The paper uses exponential decay, probably as implemented here.
        def exp_epoch_decay_sched(epoch):
            lr = self.parameters['learning_rate']
            de = self.parameters['exponential_decay']
            lr_new = lr * tf.pow(de, epoch)
            self.log.debug("epoch: %d, lr: %f, de: %f: lr_new: %f", epoch, lr, de, lr_new)
            return lr_new

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.checkpoint_callback(),
            self.keras.log_predictions(validation_images),
            tf.keras.callbacks.LearningRateScheduler(exp_epoch_decay_sched)
        ]

        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )

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
