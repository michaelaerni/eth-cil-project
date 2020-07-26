import argparse
import typing

import numpy as np

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Old Baseline (Logistic Regression)'
EXPERIMENT_TAG = 'baseline_logreg'


class BaselineLogisticRegressionExperiment(rs.framework.FitExperiment):

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
            satellite_images, groundtruth_images = rs.data.cil.load_images(training_sample_paths)
            satellite_image_patches = rs.data.cil.cut_patches(satellite_images)
            self.log.debug('Satellite image patches shape: %s', satellite_image_patches.shape)
        except OSError:
            self.log.exception('Unable to read training data')
            return

        self.log.info(
            'Loaded %d training samples consisting of %d patches each',
            satellite_image_patches.shape[0],
            satellite_image_patches.shape[1]
        )

        # Flatten training patches
        training_image_patches = flatten_patches(satellite_image_patches)

        # Extract features from training images
        training_features = extract_features(training_image_patches)
        self.log.debug('Training features shape: %s', training_features.shape)

        # Calculate labels from groundtruth
        training_labels = rs.data.cil.segmentation_to_patch_labels(groundtruth_images)
        training_labels = np.reshape(training_labels, (-1,))
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
            current_patches = flatten_patches(rs.data.cil.cut_patches(np.expand_dims(image, axis=0)))
            current_features = extract_features(current_patches)

            target_height = image.shape[0] // rs.data.cil.PATCH_SIZE
            target_width = image.shape[1] // rs.data.cil.PATCH_SIZE

            result[sample_id] = np.reshape(classifier.predict(current_features), (target_height, target_width))

        return result


def flatten_patches(patches: np.ndarray) -> np.ndarray:
    return np.reshape(
        patches,
        (-1, patches.shape[3], patches.shape[4], patches.shape[5])
    )


def extract_features(patches: np.ndarray) -> np.ndarray:
    # Pixel values are in [0, 1]
    assert np.min(patches) >= 0.0 and np.max(patches) <= 1.0

    means = np.mean(patches, axis=(1, 2))
    variances = np.var(patches, axis=(1, 2))

    return np.concatenate((means, variances), axis=1)


if __name__ == '__main__':
    BaselineLogisticRegressionExperiment().run()
