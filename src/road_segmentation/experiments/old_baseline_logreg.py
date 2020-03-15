import argparse
import logging

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Old Baseline (Logistic Regression)'
EXPERIMENT_TAG = 'baseline_logreg'


def main():
    # TODO: This is only temporary for testing
    logging.basicConfig(level=logging.DEBUG)

    # Initialize logging
    rs.util.setup_logging()
    log = logging.getLogger(__name__)

    # Handle CLI args
    parser = create_arg_parser()
    args = parser.parse_args()

    # Read input data
    training_sample_paths = rs.data.cil.training_sample_paths(args.data)
    # TODO: Process data

    # TODO: Setup logging

    # TODO: Read data

    pass


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(EXPERIMENT_DESCRIPTION)

    parser.add_argument('--data', type=str, default=rs.util.DEFAULT_DATA_DIR, help='Root data directory')
    parser.add_argument('--log', type=str, default=rs.util.DEFAULT_LOG_DIR, help='Root output directory')
    parser.add_argument('--n-train-samples', type=int, default=20, help='Number of training images')

    return parser


if __name__ == '__main__':
    main()
