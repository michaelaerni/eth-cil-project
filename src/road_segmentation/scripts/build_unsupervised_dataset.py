import argparse
import io
import logging
import os
import shutil
import threading
import time
import typing
import warnings
import zipfile

import PIL
import numpy as np
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
    for tile_idx, tile_path in enumerate(paths):
        _log.info('Processing tile %d of %d for %s', tile_idx + 1, len(paths), city)

        # Extract file id from zip file name
        raw_file_name = os.path.basename(tile_path)
        tile_id = os.path.splitext(raw_file_name)[0]

        # Extract raw tile as numpy array
        # This first extracts the entire binary blob from the zip file into memory since it seems to be faster
        # than simultaneous extraction and loading.
        with zipfile.ZipFile(tile_path, 'r') as zip_handle:
            with io.BytesIO(zip_handle.read(tile_id + '.tif')) as raw_data:
                with PIL.Image.open(raw_data) as image:
                    raw_tile_data = np.asarray(image)

        # Handle tile directory
        output_dir = os.path.join(base_output_dir, city, tile_id)
        if skip_existing_directories and os.path.exists(output_dir):
            _log.debug('Output tile directory %s already exists and is skipped', output_dir)
            continue
        os.makedirs(output_dir, exist_ok=True)

        # If the tiles are RGBIR images we only take the first three (RGB) channels
        raw_tile_data = raw_tile_data[:, :, :3]
        patches = rs.data.unsupervised.extract_patches_from_image(raw_tile_data)

        # Save the resulting patches
        for patch_idx in range(len(patches)):
            output_file = os.path.join(output_dir, str(patch_idx) + '.png')
            Image.fromarray(patches[patch_idx]).save(output_file)

        total_number_of_patches += len(patches)
    end_time = time.time() - start_time
    _log.info(
        'Finished city %s in %s with %d resulting patches', city, end_time, total_number_of_patches
    )


if __name__ == '__main__':
    main()
