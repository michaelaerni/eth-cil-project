import argparse
import typing

import matplotlib.image
import numpy as np

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Old Baseline (Logistic Regression)'
EXPERIMENT_TAG = 'baseline_logreg'



# TODO: All this stuff should be gone
# FIXME: Hardcoding this is ugly, either read dynamically or move to rs.data.cil
_BACKGROUND_THRESHOLD = 0.25



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
            (-1, rs.data.cil.PATCH_SIZE, rs.data.cil.PATCH_SIZE, satellite_image_patches.shape[-1])
        )
        training_groundtruth_patches = np.reshape(
            groundtruth_image_patches,
            (-1, rs.data.cil.PATCH_SIZE, rs.data.cil.PATCH_SIZE)
        )

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

        return classifier

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            current_patches = image_to_patches(image)
            current_features = extract_features(current_patches)

            target_height = image.shape[0] // rs.data.cil.PATCH_SIZE
            target_width = image.shape[1] // rs.data.cil.PATCH_SIZE

            result[sample_id] = np.reshape(classifier.predict(current_features), (target_height, target_width))

        return result


def main():
    BaselineLogisticRegressionExperiment().run()


def load_image_patches(path: str) -> np.ndarray:
    # Load raw data
    raw_data = matplotlib.image.imread(path)

    return image_to_patches(raw_data)


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


if __name__ == '__main__':
    main()
