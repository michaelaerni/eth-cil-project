import abc
import argparse
import csv
import datetime
import json
import logging
import os
import typing

import matplotlib.image
import numpy as np

import road_segmentation as rs

_LOG_FORMAT = '%(asctime)s  %(levelname)s [%(name)s]: %(message)s'
_PARAMETER_FILE_NAME = 'parameters.json'


# TODO: Model saving
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
                # TODO: The effective image reading belongs to data
                test_prediction_input[test_sample_id] = matplotlib.image.imread(test_sample_path)
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

                    input_size = test_prediction_input[test_sample_id].shape[:2]

                    if predicted_segmentation.shape == input_size:
                        # TODO: Implement
                        raise NotImplementedError('Segmentation to patches not implemented yet')

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


def _setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def _create_output_id(sample_id: int, patch_x: int, patch_y: int) -> str:
    x = patch_x * rs.data.cil.PATCH_SIZE
    y = patch_y * rs.data.cil.PATCH_SIZE
    return f'{sample_id:03d}_{x}_{y}'
