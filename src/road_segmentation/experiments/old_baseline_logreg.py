import argparse
import csv
import os
import typing

import matplotlib.image
import numpy as np

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Old Baseline (Logistic Regression)'
EXPERIMENT_TAG = 'baseline_logreg'



# TODO: All this stuff should be gone
# FIXME: Hardcoding this is ugly, either read dynamically or move to rs.data.cil
_PATCH_SIZE = 16
_BACKGROUND_THRESHOLD = 0.25
_TEST_IMAGE_SIZE = 608



class BaselineLogisticRegressionExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--n-train-samples', type=int, default=20, help='Maximum number of training images to use')
        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'num_training_samples': args.n_train_samples
        }

    def fit(self) -> typing.Any:
        num_training_samples = self.parameters['num_training_samples']

        # Read training data
        try:
            training_sample_paths = rs.data.cil.training_sample_paths(self.data_directory)[:num_training_samples]

            satellite_image_patches = np.asarray([load_image_patches(path) for (path, _) in training_sample_paths])
            self.log.debug('Satellite image patches shape: %s', satellite_image_patches.shape)
            groundtruth_image_patches = np.asarray([load_image_patches(path) for (_, path) in training_sample_paths])
            self.log.debug('Groundtruth image patches shape: %s', groundtruth_image_patches.shape)
        except OSError:
            self.log.exception('Unable to read training data')
            return

        assert satellite_image_patches.shape[1] == groundtruth_image_patches.shape[1]
        self.log.info(
            'Loaded %d training samples consisting of %d patches each',
            satellite_image_patches.shape[0],
            satellite_image_patches.shape[1]
        )

        # Flatten training patches
        training_image_patches = np.reshape(
            satellite_image_patches,
            (-1, _PATCH_SIZE, _PATCH_SIZE, satellite_image_patches.shape[-1])
        )
        training_groundtruth_patches = np.reshape(groundtruth_image_patches, (-1, _PATCH_SIZE, _PATCH_SIZE))

        # Extract features from training images
        training_features = extract_features(training_image_patches)
        self.log.debug('Training features shape: %s', training_features.shape)

        # Calculate labels from groundtruth
        training_labels = calculate_labels(training_groundtruth_patches)
        self.log.debug('Training labels shape: %s', training_labels.shape)
        self.log.info(
            'Using %d background and %d foreground patches for training',
            np.sum(1.0 - training_labels),
            np.sum(training_labels)
        )

        assert training_features.shape[0] == training_labels.shape[0]

        # TODO: The counts do not quite match those from the old baseline Jupyter notebook

        # Fit classifier
        classifier = rs.models.baseline.create_old_logreg_model(self.SEED)
        self.log.info('Fitting classifier')
        classifier.fit(training_features, training_labels)
        self.log.info('Classifier fitted')

        # TODO: Prediction also belongs into Experiment class

        # Predict on test data
        self.log.info('Predicting test data')
        output_file = os.path.join(self.experiment_directory, 'submission.csv')
        try:
            test_samples = rs.data.cil.test_sample_paths(self.data_directory)

            with open(output_file, 'w', newline='') as f:
                # Write CSV header
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Id', 'Prediction'])
                for test_sample_id, test_sample_path in test_samples:
                    self.log.debug('Predicting sample %d from %s', test_sample_id, test_sample_path)
                    current_patches = load_image_patches(test_sample_path)
                    current_features = extract_features(current_patches)
                    current_predictions = classifier.predict(current_features)

                    for y in range(0, _TEST_IMAGE_SIZE, _PATCH_SIZE):
                        for x in range(0, _TEST_IMAGE_SIZE, _PATCH_SIZE):
                            prediction_idx = (y // _PATCH_SIZE) * (_TEST_IMAGE_SIZE // _PATCH_SIZE) + (x // _PATCH_SIZE)
                            output_id = create_output_id(test_sample_id, x, y)
                            writer.writerow([output_id, current_predictions[prediction_idx]])
        except OSError:
            self.log.exception('Unable to read test data')
            return

        self.log.info('Saved predictions to %s', output_file)

        return classifier


def main():
    BaselineLogisticRegressionExperiment().run()


def load_image_patches(path: str) -> np.ndarray:
    # Load raw data
    raw_data = matplotlib.image.imread(path)

    # Reshape and rearrange axes
    reshaped = np.reshape(
        raw_data,
        (raw_data.shape[0] // _PATCH_SIZE, _PATCH_SIZE, raw_data.shape[1] // _PATCH_SIZE, _PATCH_SIZE, -1)
    )
    rearranged = np.swapaxes(reshaped, 1, 2)

    # Flatten 2D grid of patches
    flattened = np.reshape(rearranged, (-1, _PATCH_SIZE, _PATCH_SIZE, rearranged.shape[-1]))

    return flattened


def extract_features(patches: np.ndarray) -> np.ndarray:
    # Pixel values are in [0, 1]
    assert np.min(patches) >= 0.0 and np.max(patches) <= 1.0

    means = np.mean(patches, axis=(1, 2))
    variances = np.var(patches, axis=(1, 2))

    return np.concatenate((means, variances), axis=1)


def calculate_labels(groundtruth_patches: np.ndarray) -> np.ndarray:
    # Pixel values are in {0, 1}
    foreground = np.mean(groundtruth_patches, axis=(1, 2)) > _BACKGROUND_THRESHOLD
    return foreground.astype(np.int)


def create_output_id(sample_id: int, x: int, y: int) -> str:
    return f'{sample_id:03d}_{x}_{y}'


if __name__ == '__main__':
    main()
