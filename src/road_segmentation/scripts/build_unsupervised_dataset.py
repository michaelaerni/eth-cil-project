import argparse
import logging
import os
import shutil
import threading
import time
import typing
import warnings

from PIL import Image

import road_segmentation as rs

_LOG_FORMAT = '%(asctime)s  %(levelname)s [%(name)s]: %(message)s'

_log = logging.getLogger(__name__)


def main():
    # Prepare args and logging
    args = _create_arg_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format=_LOG_FORMAT)

    # Disables warnings about image decompression bombs.
    # If this is not present then some images might be falsely classified as a bomb since
    # their extracted size is quite large.
    # This requires carefully examining the acquired raw data set!
    _log.warning('Disabling warnings for image decompression bombs. Make sure your data is trustworthy!')
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    # Prepare output directory
    base_output_dir = os.path.join(args.datadir, 'processed', rs.data.unsupervised.DATASET_TAG)
    _log.info('Using output directory %s', base_output_dir)
    if not args.skip_existing:
        # Delete complete output directory if recreating from scratch
        shutil.rmtree(base_output_dir, ignore_errors=True)
    os.makedirs(base_output_dir)

    # Load raw image paths
    _log.info('Using base data directory directory %s', args.datadir)
    try:
        tile_paths_per_city = rs.data.unsupervised.raw_data_paths(args.datadir)
    except FileNotFoundError:
        _log.exception('Input directories could not be found. Is the input directory structure correct?')
        return

    # Process cities in parallel
    for city, paths in tile_paths_per_city.items():
        # FIXME: Could use multiprocessing for easier use
        threading.Thread(
            target=_process_city,
            args=(paths, city, base_output_dir, args.skip_existing)
        ).start()


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Process raw NAIP TIF images into the unsupervised data set')

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip existing tiles. If not set then all patches will be created from scratch.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument('--datadir', type=str, default=rs.util.DEFAULT_DATA_DIR, help='Root data directory')

    return parser


def _process_city(
        paths: typing.List[str],
        city: str,
        base_output_dir: str,
        skip_existing_directories: bool
):
    """
    Processes a city, i.e. extract patches for all tiles of that city
    """

    start_time = time.time()

    total_number_of_patches = 0
    for i, tile_path in enumerate(paths):
        _log.info('Processing tile %d of %d for %s', i + 1, len(paths), city)
        tile_name = tile_path.split('/')[-1][:-4]
        output_dir = os.path.join(base_output_dir, city, tile_name)
        if skip_existing_directories and os.path.exists(output_dir):
            _log.info(f'Skip: {tile_name}')
            continue
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)

        image = rs.data.cil.load_image(tile_path)
        image = image[:, :, :3]
        patches = rs.data.unsupervised.extract_patches_from_image(image)
        for idx in range(len(patches)):
            output_file = os.path.join(output_dir, str(idx) + '.png')
            Image.fromarray(patches[idx]).save(output_file)

        total_number_of_patches += len(patches)
    end_time = time.time() - start_time
    _log.info(
        'Finished city %s in %s with %d resulting patches', city, end_time, total_number_of_patches
    )


if __name__ == '__main__':
    main()
