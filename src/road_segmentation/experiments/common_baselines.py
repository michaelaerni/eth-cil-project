import argparse
import enum
import typing

import numpy as np

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Common Baselines'
EXPERIMENT_TAG = 'common_baselines'


class BaselineType(enum.Enum):
    """
    Type of baseline to use.
    """

    NULL = 'null'
    """
    Predict always background (0).
    """
    ONE = 'one'
    """
    Predict always foreground (1).
    """
    RANDOM_UNIFORM = 'uniform'
    """
    Predict randomly with uniform probability.
    """
    RANDOM_PRIOR = 'prior'
    """
    Predict randomly with prior class label probability.
    """


class DummyBaselineExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            '--type',
            type=str,
            required=True,
            choices=list(map(lambda e: e.value, BaselineType)),
            help='Type of dummy baseline'
        )
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'type': args.type
        }

    def fit(self) -> typing.Any:

        baseline_type = BaselineType(self.parameters['type'])
        self.log.info('Using %s baseline', baseline_type.name)

        if baseline_type != BaselineType.RANDOM_PRIOR:
            # No need to fit anything at all
            return None
        else:
            # Need to calculate label prior
            self.log.debug('Calculating label prior')
            try:
                training_paths, _ = rs.data.cil.train_validation_sample_paths(self.data_directory)
                _, training_masks = rs.data.cil.load_images(training_paths)
                self.log.debug('Loaded %d training masks', training_masks.shape[0])
            except (OSError, ValueError):
                self.log.exception('Unable to load training data')
                return

            # Use patch labels for prior since they are more well-defined
            training_masks = rs.data.cil.segmentation_to_patch_labels(training_masks)
            assert np.alltrue((training_masks == 0.0) | (training_masks == 1.0))

            # Prior is simply mean of labels
            return np.mean(training_masks)

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        prediction_function = self._prediction_function(classifier)
        return {
            sample_id: prediction_function(image) for sample_id, image in images.items()
        }

    def _prediction_function(self, classifier: typing.Optional[np.float]) -> typing.Callable[[np.ndarray], np.ndarray]:
        baseline_type = BaselineType(self.parameters['type'])

        if baseline_type == BaselineType.RANDOM_PRIOR:
            self.log.info('Label prior probability: %.4f', classifier)

        random = np.random.RandomState(self.SEED)

        return {
            BaselineType.NULL: lambda image: self._predict_null(self._output_shape(image)),
            BaselineType.ONE: lambda image: self._predict_one(self._output_shape(image)),
            BaselineType.RANDOM_UNIFORM: lambda image: self._predict_random(self._output_shape(image), 0.5, random),
            BaselineType.RANDOM_PRIOR: lambda image: self._predict_random(self._output_shape(image), classifier, random)
        }[baseline_type]

    @classmethod
    def _output_shape(cls, image: np.ndarray) -> typing.Tuple[int, int]:
        return image.shape[0] // rs.data.cil.PATCH_SIZE, image.shape[1] // rs.data.cil.PATCH_SIZE

    @classmethod
    def _predict_null(cls, shape: typing.Tuple[int, int]) -> np.ndarray:
        return np.zeros(shape, dtype=np.int)

    @classmethod
    def _predict_one(cls, shape: typing.Tuple[int, int]) -> np.ndarray:
        return np.ones(shape, dtype=np.int)

    @classmethod
    def _predict_random(cls, shape: typing.Tuple[int, int], prior_probability: float, random: np.random.RandomState) -> np.ndarray:
        assert prior_probability is not None
        return (random.uniform(size=shape) < prior_probability).astype(np.int)


def main():
    DummyBaselineExperiment().run()


if __name__ == '__main__':
    main()
