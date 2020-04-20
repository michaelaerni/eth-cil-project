import argparse
import typing

import matplotlib.image
import numpy as np

import road_segmentation as rs

import tensorflow as tf
import matplotlib.pyplot as plt

EXPERIMENT_DESCRIPTION = 'U-Net Baseline'
EXPERIMENT_TAG = 'baseline_unet'


def pad_2D(images, padding):
    return np.asarray(
        [np.pad(x, ((padding, padding), (padding, padding), (0, 0)), mode="symmetric") for x in images])


class BaselineUnetExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
        parser.add_argument('--momentum', type=float, default=0.99, help='Momentum')
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        # TODO: Evaluation
        # TODO: Data augmentation

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

        padded_training_images = pad_2D(training_images, 94)
        padded_validation_images = pad_2D(validation_images, 94)

        padded_training_masks = pad_2D(training_masks, 2)
        padded_validation_masks = pad_2D(validation_masks, 2)

        training_dataset = tf.data.Dataset.from_tensor_slices((padded_training_images, padded_training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=1024)
        training_dataset = training_dataset.batch(batch_size)
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        validation_dataset = tf.data.Dataset.from_tensor_slices((padded_validation_images, padded_validation_masks))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        model = rs.models.unet.UNet()
        sgd_optimizer = tf.keras.optimizers.SGD(momentum=self.parameters['momentum'],
                                                learning_rate=self.parameters['learning_rate'])
        model.compile(optimizer=sgd_optimizer,
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=[
                          'accuracy'
                      ])
        tf.keras.utils.plot_model(model, show_shapes=True, )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.checkpoint_callback(),
            self.keras.log_predictions(padded_validation_images)
        ]

        # Fit model
        model_history = model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )

        plot_losses(model_history, self.parameters['epochs'])

        """
        implemented u-net like in paper, but other shapes
        
        need to implement mirroring, right now just padding
        """

        image = np.asarray([padded_training_images[0]])
        predictions = model.predict(image)
        predictions = crop_center(predictions, 400, 400)
        prediction_mask = np.argmax(predictions[0], -1)

        display([training_images[0], training_masks[0].reshape((400, 400, 1)),
                 prediction_mask.reshape((400, 400, 1))])

        self.log.info('Classifier fitted')

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)

            target_height = image.shape[0]
            target_width = image.shape[1]

            # todo: mirroring instead of just padding
            image = np.pad(image, ((94, 94), (94, 94), (0, 0)), mode="symmetric")
            image = np.asarray([image])

            predictions = classifier.predict(image)
            predictions = crop_center(predictions, target_height, target_width)
            prediction_mask = np.argmax(predictions[0], -1)

            prediction_mask_patches = image_to_patches(prediction_mask)
            labels = np.asarray([patch_to_label(patch) for patch in prediction_mask_patches])

            target_height = target_height // rs.data.cil.PATCH_SIZE
            target_width = target_width // rs.data.cil.PATCH_SIZE

            labels = np.reshape(labels, (target_height, target_width))
            result[sample_id] = labels

            assert result[sample_id].shape == (target_height, target_width)

        return result


def main():
    BaselineUnetExperiment().run()


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def load_image(path: str) -> np.ndarray:
    # Load raw data
    return matplotlib.image.imread(path)


def image_to_patches(raw_image: np.ndarray) -> np.ndarray:
    # Reshape and rearrange axes
    reshaped = np.reshape(
        raw_image,
        (
            raw_image.shape[0] // rs.data.cil.PATCH_SIZE,
            rs.data.cil.PATCH_SIZE, raw_image.shape[1] // rs.data.cil.PATCH_SIZE,
            rs.data.cil.PATCH_SIZE,
            -1
        )
    )
    rearranged = np.swapaxes(reshaped, 1, 2)
    # Flatten 2D grid of patches
    flattened = np.reshape(rearranged, (-1, rs.data.cil.PATCH_SIZE, rs.data.cil.PATCH_SIZE, rearranged.shape[-1]))

    return flattened


def calculate_labels(groundtruth_images: np.ndarray) -> np.ndarray:
    return np.where(groundtruth_images < 0.5, 0, 1)


def plot_losses(model_history, epochs):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure()
    plt.plot(epochs_range, loss, 'r', label='Training loss')
    plt.plot(epochs_range, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


# assign a label to a patch
def patch_to_label(patch):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def crop_center(image, target_height, target_width):
    _, y, x, _ = image.shape
    startx = x // 2 - (target_width // 2)
    starty = y // 2 - (target_height // 2)
    return image[:, starty:starty + target_height, startx:startx + target_width, :]


if __name__ == '__main__':
    main()
