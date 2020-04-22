import abc
import argparse
import csv
import datetime
import json
import logging
import os
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

_LOG_FORMAT = '%(asctime)s  %(levelname)s [%(name)s]: %(message)s'
_PARAMETER_FILE_NAME = 'parameters.json'


# TODO: Model restoring
# TODO: Running only evaluation or prediction
# TODO: Evaluation


class Experiment(metaclass=abc.ABCMeta):
    """
    Abstract base class to be extended by concrete experiments.
    """

    SEED = 42
    """
    Random seed to be used for consistent RNG initialisation.
    """

    @property
    @abc.abstractmethod
    def tag(self) -> str:
        """
        Returns:
            Experiment tag
        """
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """
        Returns:
            Human-readable experiment description
        """
        pass

    @abc.abstractmethod
    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Create an argument parser which handles
        arguments specific to this experiment.
        A template parser to be used is given as a parameter.
        General arguments (e.g. debug flag) are added automatically.

        Args:
            parser: Template parser to be extended and returned.

        Returns:
            Argument parser for parsing experiment-specific arguments.

        """
        pass

    @abc.abstractmethod
    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        """
        Collect all experimental parameters into a dictionary.
        The result must be sufficient to exactly and completely
        reproduce the experiment.
        Parameters independent of the experiment are added automatically.

        Keys must not start with `base_`.

        Args:
            args: Parsed command line arguments.

        Returns:
            Dictionary where keys are strings, indicating experimental parameters.

        """
        pass

    @abc.abstractmethod
    def fit(self) -> typing.Any:
        """
        Fit a classifier for the current experiment.

        Returns:
            Fitted classifier and other objects required for prediction and evaluation.
        """
        pass

    @abc.abstractmethod
    def predict(
            self,
            classifier: typing.Any,
            images: typing.Dict[int, np.ndarray]
    ) -> typing.Dict[int, np.ndarray]:
        """
        Run a fitted classifier on a list of images and return their segmentations.
        The resulting segmentations should either have the same resolution
        as the input images or be reduced by the target patch size.

        Args:
            classifier: Fitted classifier and other objects as returned by fit.
            images: Images to run prediction on.
             Keys are ids and values the actual images.
             Each image is of shape H x W x 3, with H and W being multiples of patch size.

        Returns:
            Predicted segmentation masks.
             Keys are ids and values the actual images.
             Each image must be either of shape H x W or (H / patch size) x (W / patch size).
        """
        pass

    def __init__(self):
        self._parameters = None
        self._log = None
        self._experiment_directory = None
        self._keras_helper = None

    def run(self):
        # Fix seeds as a failsafe (as early as possible)
        rs.util.fix_seeds(self.SEED)

        # Parse CLI args
        parser = self._create_full_argument_parser()
        args = parser.parse_args()

        # Collect all experimental parameters in a dictionary for reproducibility
        self._parameters = self._build_parameter_dict(args)

        # Initialise logging
        _setup_logging(debug=self.parameters['base_is_debug'])
        self._log = logging.getLogger(__name__)

        self.log.debug('Experiment parameters: %s', self.parameters)

        # Initialise experiment directory
        directory_name = f'{self.tag}_{datetime.datetime.now():%y%m%d-%H%M%S}'
        self._experiment_directory = os.path.join(self.parameters['base_log_directory'], directory_name)
        try:
            os.makedirs(self._experiment_directory, exist_ok=False)
        except OSError:
            self.log.exception('Unable to setup output directory')
            return
        self.log.info('Using experiment directory %s', self._experiment_directory)

        # Initialise helpers
        self._keras_helper = KerasHelper(self.experiment_directory)

        # Store experiment parameters
        # noinspection PyBroadException
        try:
            with open(os.path.join(self._experiment_directory, _PARAMETER_FILE_NAME), 'w') as f:
                json.dump(self.parameters, f)
        except Exception:
            self.log.exception('Unable to save parameters')
            return

        # Fit model
        classifier = self.fit()

        # TODO: Evaluation

        # Predict on test data
        self.log.info('Predicting test data')
        output_file = os.path.join(self.experiment_directory, 'submission.csv')
        try:
            self.log.debug('Reading test inputs')
            test_prediction_input = dict()
            for test_sample_id, test_sample_path in rs.data.cil.test_sample_paths(self.data_directory):
                test_prediction_input[test_sample_id] = rs.data.cil.load_image(test_sample_path)
        except OSError:
            self.log.exception('Unable to read test data')
            return

        self.log.debug('Running classifier on test data')
        test_prediction = self.predict(classifier, test_prediction_input)

        try:
            self.log.debug('Creating submission file')
            with open(output_file, 'w', newline='') as f:
                # Write CSV header
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Id', 'Prediction'])

                for test_sample_id, predicted_segmentation in test_prediction.items():
                    predicted_segmentation = np.squeeze(predicted_segmentation)
                    if len(predicted_segmentation.shape) != 2:
                        raise ValueError(
                            f'Expected 2D prediction (after squeeze) but got shape {predicted_segmentation.shape}'
                        )

                    # Make sure result is integer values
                    predicted_segmentation = predicted_segmentation.astype(np.int)

                    input_size = test_prediction_input[test_sample_id].shape[:2]
                    if predicted_segmentation.shape == input_size:
                        self.log.warning(
                            'Predicted test segmentation has the same size as the input images (%s). '
                            'Ideally, classifiers should perform postprocessing themselves!',
                            predicted_segmentation.shape
                        )

                        # Convert to patches (in the default way)
                        predicted_segmentation = rs.data.cil.segmentation_to_patch_labels(
                            np.expand_dims(predicted_segmentation, axis=(0, 3))
                        )[0].astype(np.int)

                    for patch_y in range(0, predicted_segmentation.shape[0]):
                        for patch_x in range(0, predicted_segmentation.shape[1]):
                            output_id = _create_output_id(test_sample_id, patch_x, patch_y)
                            writer.writerow([output_id, predicted_segmentation[patch_y, patch_x]])
        except OSError:
            self.log.exception('Unable to write submission data')
            return

        self.log.info('Saved predictions to %s', output_file)

    @property
    def log(self) -> logging.Logger:
        """
        Returns:
            Logger for the current experiment
        """
        # TODO: The logger tag is always framework, not the actual experiment. Could split that up to have correct src.
        return self._log

    @property
    def parameters(self) -> typing.Dict[str, typing.Any]:
        """
        Returns:
            Experiment parameters.
        """
        return self._parameters

    @property
    def data_directory(self):
        """
        Returns:
            Root data directory.
        """
        return self.parameters['base_data_directory']

    @property
    def experiment_directory(self):
        """
        Returns:
            Current experiment output directory.
        """
        return self._experiment_directory

    @property
    def keras(self) -> 'KerasHelper':
        """
        Returns:
            Keras helper.
        """
        return self._keras_helper

    def _create_full_argument_parser(self) -> argparse.ArgumentParser:
        # Create template parser
        parser = argparse.ArgumentParser(self.description)

        # Add general arguments
        parser.add_argument('--datadir', type=str, default=rs.util.DEFAULT_DATA_DIR, help='Root data directory')
        parser.add_argument('--logdir', type=str, default=rs.util.DEFAULT_LOG_DIR, help='Root output directory')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')

        # Add experiment-specific arguments
        parser = self.create_argument_parser(parser)
        return parser

    def _build_parameter_dict(self, args):
        # Validate parameters from child class
        parameters = self.build_parameter_dict(args)
        if any(filter(lambda key: key is not None and str(key).startswith('base_'), parameters.keys())):
            raise ValueError('Parameter keys must not start with `base_`')

        # Add general parameters
        parameters['base_data_directory'] = args.datadir
        parameters['base_log_directory'] = args.logdir
        parameters['base_is_debug'] = args.debug

        return parameters


class KerasHelper(object):
    # TODO: Document this class

    def __init__(self, log_dir: str):
        """
        Create a new keras helper.
        Args:
            log_dir: Root log directory of the current experiment.
        """
        self._log_dir = log_dir
        self._log = logging.getLogger(__name__)

    def tensorboard_callback(self) -> tf.keras.callbacks.Callback:
        self._log.info(
            'Setting up tensorboard logging, use with `tensorboard --logdir=%s`', self._log_dir
        )
        return tf.keras.callbacks.TensorBoard(
            self._log_dir,
            histogram_freq=10,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        )

    def checkpoint_callback(self, period: int = 50) -> tf.keras.callbacks.Callback:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self._log_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=False)

        # Create path template
        path_template = os.path.join(checkpoint_dir, '{epoch:04d}-{val_loss:.4f}.h5')

        return tf.keras.callbacks.ModelCheckpoint(
            path_template,
            save_best_only=False,
            period=period  # TODO: This API is deprecated, however, there does not seem to be a way to get the same results
        )

    def log_predictions(
            self,
            validation_images: np.ndarray,
            freq: int = 10
    ) -> tf.keras.callbacks.Callback:
        return self._LogPredictionsCallback(
            os.path.join(self._log_dir, 'validation_predictions'),
            validation_images,
            freq
        )

    def default_metrics(self, threshold: float) -> typing.List[tf.keras.metrics.Metric]:
        # TODO: Check how this works w.r.t. batching and validation. Actual results might be skewed!

        return [
            rs.metrics.BinaryMeanFScore(threshold=threshold),
            rs.metrics.BinaryMeanIoUScore(threshold=threshold)
        ]

    class _LogPredictionsCallback(tf.keras.callbacks.LambdaCallback):
        def __init__(
                self,
                log_dir: str,
                validation_images: np.ndarray,
                freq: int
        ):
            super().__init__(on_epoch_end=lambda epoch, _: self._log_predictions_callback(epoch))

            self._writer = tf.summary.create_file_writer(log_dir)
            self._validation_images = validation_images
            self._freq = freq
            self._model: typing.Optional[tf.keras.Model] = None

        def set_model(self, model: tf.keras.Model):
            self._model = model

        def _log_predictions_callback(
                self,
                epoch: int
        ):
            if epoch % self._freq != 0:
                return

            assert self._model is not None

            # Predict segmentations
            segmentations = self._model.predict(self._validation_images)

            # Prediction are logits, thus convert into correct range
            segmentations = tf.sigmoid(segmentations)

            # Create overlay images
            if self._validation_images.shape[1:3] != segmentations.shape[1:3]:
                # Rescale segmentations if necessary
                scaled_segmentations = tf.image.resize(
                    segmentations,
                    size=self._validation_images.shape[1:3],
                    method='nearest'
                )
            else:
                scaled_segmentations = segmentations
            mask_strength = 0.7
            overlay_images = mask_strength * scaled_segmentations * self._validation_images + (1.0 - mask_strength) * self._validation_images
            with self._writer.as_default():
                tf.summary.image('predictions_overlay', overlay_images, step=epoch, max_outputs=overlay_images.shape[0])
                tf.summary.image('predictions', segmentations, step=epoch, max_outputs=segmentations.shape[0])


def _setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def _create_output_id(sample_id: int, patch_x: int, patch_y: int) -> str:
    x = patch_x * rs.data.cil.PATCH_SIZE
    y = patch_y * rs.data.cil.PATCH_SIZE
    return f'{sample_id:03d}_{x}_{y}'
