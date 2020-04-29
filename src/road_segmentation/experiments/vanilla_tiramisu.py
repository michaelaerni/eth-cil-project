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
        parser.add_argument('--batch-size', type=int, default=3, help='Training batch size.')
        parser.add_argument('--dropout-rate', type=float, default=0.2, help='Dropout rate.')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate.')
        parser.add_argument('--learning-rate-finetune', type=float, default=1e-4,
                            help='Learning rate for the restarted training session.')
        parser.add_argument('--exponential-decay', type=float, default=0.995,
                            help='Exponential decay for learning rate after each epoch.')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Factor for weight decay regularisers.')
        parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs.')
        parser.add_argument('--patience', type=int, default=100, help='Patience for the initial training.')
        parser.add_argument('--patience-finetune', type=int, default=50,
                            help='Patience for the restarted training session.')
        parser.add_argument('--patience-metric', type=str, default='val_binary_mean_iou_score',
                            choices=['binary_mean_iou_score', 'binary_mean_accuracy', 'binary_mean_f_score',
                                     'val_loss', 'val_binary_mean_accuracy', 'val_binary_mean_f_score',
                                     'val_binary_mean_iou_score', 'loss'],
                            help='Metric to be used by patience early stopping.')
        parser.add_argument('--fcnet', type=int, default=103, choices=[0, 56, 67, 103],
                            help='Specify model to be used. 0: tiny testing net, 56 / 67 / 103: FC-DenseNet56/67/103 respectively.')
        parser.add_argument('--n-layers-per-dense-block', type=int, nargs='+', default=[],
                            help='Define the number of layers in each dense blocks of the tiramisu.')
        parser.add_argument('--no-mirror-dense-blocks', dest='mirror_dense_blocks', default=True, action='store_false',
                            help='If given, the model dense blocks will be interpreted as te full specification of the dense blocks in the tiramisu instead of a just the down path plus bottleneck.')
        parser.add_argument('--growth-rate', type=int, default=16, help='Growth rate in of dense blocks.')
        parser.add_argument('--n-initial-features', type=int, default=48,
                            help='Number of feature maps of the first convolutional layer in the tiramisu.')
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
            'patience': args.patience,
            'patience_finetune': args.patience_finetune,
            'patience_metric': args.patience_metric,
            'fcnet': args.fcnet,
            'n_layers_per_dense_block': args.n_layers_per_dense_block,
            'mirror_dense_blocks': args.mirror_dense_blocks,
            'growth_rate': args.growth_rate,
            'n_initial_features': args.n_initial_features
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']
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

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(batch_size)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        if len(self.parameters['n_layers_per_dense_block']) == 0:
            fcnet = self.parameters['fcnet']
            if fcnet == 0:
                model = rs.models.tiramisu.build_FCDenseNetTiny(dropout_rate=self.parameters['dropout_rate'],
                                                                weight_decay=self.parameters['weight_decay'])
            elif fcnet == 56:
                model = rs.models.tiramisu.build_FCDenseNet56(dropout_rate=self.parameters['dropout_rate'],
                                                              weight_decay=self.parameters['weight_decay'])
            elif fcnet == 67:
                model = rs.models.tiramisu.build_FCDenseNet67(dropout_rate=self.parameters['dropout_rate'],
                                                              weight_decay=self.parameters['weight_decay'])
            elif fcnet == 103:
                model = rs.models.tiramisu.build_FCDenseNet103(dropout_rate=self.parameters['dropout_rate'],
                                                               weight_decay=self.parameters['weight_decay'])
        else:
            model = rs.models.tiramisu.Tiramisu(n_classes=1,
                                                n_initial_features=self.parameters['n_initial_features'],
                                                n_layers_per_dense_block=self.parameters['n_layers_per_dense_block'],
                                                mirror_dense_blocks=self.parameters['mirror_dense_blocks'],
                                                growth_rate=self.parameters['growth_rate'],
                                                dropout_rate=self.parameters['dropout_rate'],
                                                weight_decay=self.parameters['weight_decay'])

        # The paper uses exponential decay, probably as implemented here.
        initial_epoch = 0
        lr = self.parameters['learning_rate']
        def exp_epoch_decay_sched(epoch):
            de = self.parameters['exponential_decay']
            lr_new = lr * tf.pow(de, epoch-initial_epoch)
            self.log.debug("epoch: %d, lr: %f, de: %f: lr_new: %f", epoch, lr, de, lr_new)
            return lr_new

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.log_predictions(validation_images),
            self.keras.best_checkpoint_callback(),
            tf.keras.callbacks.LearningRateScheduler(exp_epoch_decay_sched),
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
            callbacks=callbacks + [
                tf.keras.callbacks.EarlyStopping(monitor=self.parameters['patience_metric'],
                                                 min_delta=0,
                                                 patience=self.parameters['patience'],
                                                 mode=patience_mode)
            ]
        )

        self.log.info("Loading best model")

        model.load_weights(self.keras.default_best_checkpoint_path())
        initial_epoch = len(hist.epoch)
        lr = self.parameters['learning_rate_finetune']

        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.parameters['learning_rate_finetune']),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=self.keras.default_metrics(threshold=0.0),
        )

        self.log.info("Starting fine tuning")

        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks + [
                tf.keras.callbacks.EarlyStopping(monitor=self.parameters['patience_metric'],
                                                 min_delta=0,
                                                 patience=self.parameters['patience_finetune'],
                                                 mode=patience_mode)
            ],
            initial_epoch=len(hist.epoch)
        )

        self.log.info("Loading best model")
        model.load_weights(self.keras.default_best_checkpoint_path())

        self.log.info("Training done.")

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
