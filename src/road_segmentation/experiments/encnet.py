import argparse
import typing

import numpy as np

import road_segmentation as rs
import tensorflow as tf

EXPERIMENT_DESCRIPTION = 'Context Encoding Experiment'
EXPERIMENT_TAG = 'encnet_experiment'


class ContextEncodingModuleExperient(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            '--batch-size',
            type=int,
            default=5,
            help='Training batch size.'
        )
        parser.add_argument(
            '--learning-rate',
            type=float,
            default=1e-3,
            help='Learning rate.'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=300,
            help='Number of training epochs before training stops'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training and validation data')

        try:
            training_paths, validation_paths = rs.data.cil.train_validation_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(training_paths)
            validation_images, validation_masks = rs.data.cil.load_images(validation_paths)

            self.log.debug(
                'Loaded %d training and %d validation samples',
                training_images[0],
                validation_images
            )

        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.map(dataset_add_seloss_target)
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(self.parameters['batch_size'])

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.map(dataset_add_seloss_target)
        validation_dataset = validation_dataset.batch(1)

        self.log.debug('Trainig data specification: %s', training_dataset.element_spec)
        self.log.debug('Validation data specification: %s', validation_dataset.element_spec)

        model = rs.models.encnet.EncNet(
            classes=1,
            down_path_length=1,
            codewords=12
        )

        loss_dict = {
            'output_1': tf.keras.losses.BinaryCrossentropy(from_logits=True),
            'output_2': tf.keras.losses.BinaryCrossentropy()
        }

        metrics_dict = {
            'output_1': self.keras.default_metrics(threshold=0.0)
        }

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss=loss_dict,
            metrics=metrics_dict
        )

        self.log.info("Starting training")

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.log_predictions(validation_images, prediction_idx=0),
            self.keras.best_checkpoint_callback(metric='val_output_1_binary_mean_f_score')
        ]

        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )

        self.log.info("Training done. Loading best model.")

        model.load_weights(self.keras.default_best_checkpoint_path())

        return model

    def predict(
            self,
            classifier: typing.Any,
            images: typing.Dict[int, np.ndarray]
    ) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            raw_prediction = classifier.predict(image)[0]
            prediction_mask = np.where(raw_prediction >= 0, 1, 0)
            prediction_mask_patches = rs.data.cil.segmentation_to_patch_labels(prediction_mask)

            result[sample_id] = prediction_mask_patches[0]

        return result


def dataset_add_seloss_target(image, mask):
    mask_shape = tf.shape(mask)
    y_SE_loss = tf.reduce_sum(tf.cast(mask, tf.float32)) / tf.cast((mask_shape[0] * mask_shape[1]), tf.float32)
    return image, (mask, y_SE_loss)


def main():
    ContextEncodingModuleExperient().run()


if __name__ == '__main__':
    main()
