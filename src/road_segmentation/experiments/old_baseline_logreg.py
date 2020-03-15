import argparse
import logging

import matplotlib.image
import numpy as np

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Old Baseline (Logistic Regression)'
EXPERIMENT_TAG = 'baseline_logreg'


_PATCH_SIZE = 16


def main():
    # Handle CLI args
    parser = create_arg_parser()
    args = parser.parse_args()

    data_directory = args.data
    log_directory = args.log
    num_training_samples = args.n_train_samples
    is_debug = args.debug

    # Initialize logging
    rs.util.setup_logging(debug=is_debug)
    log = logging.getLogger(__name__)

    # Initialize experiment
    try:
        experiment_dir = rs.util.setup_experiment(log_directory, EXPERIMENT_TAG)
        log.info('Using experiment directory %s', experiment_dir)
    except OSError as ex:
        log.exception('Unable to setup output directory')
        return

    # Read input data
    try:
        training_sample_paths = rs.data.cil.training_sample_paths(data_directory)[:num_training_samples]

        satellite_image_patches = np.asarray([load_image_patches(path) for (path, _) in training_sample_paths])
        log.debug('Satellite image patches shape: %s', satellite_image_patches.shape)
        groundtruth_image_patches = np.asarray([load_image_patches(path) for (_, path) in training_sample_paths])
        log.debug('Groundtruth image patches shape: %s', groundtruth_image_patches.shape)
    except OSError as ex:
        log.exception('Unable to read training data')
        return

    assert satellite_image_patches.shape[1] == groundtruth_image_patches.shape[1]
    log.info(
        'Loaded %d training samples consisting of %d patches each',
        satellite_image_patches.shape[0],
        satellite_image_patches.shape[1]
    )

    # TODO: Run experiment


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(EXPERIMENT_DESCRIPTION)

    parser.add_argument('--data', type=str, default=rs.util.DEFAULT_DATA_DIR, help='Root data directory')
    parser.add_argument('--log', type=str, default=rs.util.DEFAULT_LOG_DIR, help='Root output directory')
    parser.add_argument('--n-train-samples', type=int, default=20, help='Maximum number of training images to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    return parser


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


if __name__ == '__main__':
    main()
