import os
import threading
import time
import typing
import warnings

from PIL import Image

import road_segmentation as rs

# FIXME: figure out where to document this
"""
This script process the unsupervised data.
First it loads the .tif files and extract patches, which are then stored as .png files
in a separate directory for each city and tile.

Each city is processed in a new thread
"""


def process_city(tile_paths_per_city: typing.Dict[str, typing.List[str]],
                 city: str,
                 base_output_dir: str,
                 skip_existing_directories: bool):
    """
    Processes a city, i.e. extract patches for all tiles of that city
    """
    print('Processing {}... (Takes a few minutes)'.format(city))
    start_time = time.time()

    total_number_of_patches = 0
    for i, tile_path in enumerate(tile_paths_per_city[city]):
        print('Tile {} of {} for {}'.format(i + 1, len(tile_paths_per_city[city]), city))
        tile_name = tile_path.split('/')[-1][:-4]
        output_dir = os.path.join(base_output_dir, city, tile_name)
        if skip_existing_directories and os.path.exists(output_dir):
            print(f'Skip: {tile_name}')
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
    print(f'Number of patches for {city}: {total_number_of_patches}\n' +
          f'Processing took {end_time} seconds')


def preprocess_unsupervised_data(data_dir: str = None,
                                 skip_existing_directories: bool = False):
    """
    Main method to run unsupervised data preprocessing.
    Extracts patches for each city and each tile into a separate directory.
    Each city is processed in a new thread
    Args:
        data_dir: In case data directory is different from default
        skip_existing_directories: If true then only tiles for which no output directory exists are processed
    """
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)

    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    tile_paths_per_city = rs.data.unsupervised.raw_data_paths(data_dir)
    base_output_dir = os.path.join(data_dir, 'processed', 'unsupervised')

    for city in rs.data.unsupervised.CITIES:
        threading.Thread(target=process_city,
                         args=(tile_paths_per_city,
                               city,
                               base_output_dir,
                               skip_existing_directories)).start()


def main():
    skip_existing_directories = True  # If true then only tiles for which no output directory exists are processed
    preprocess_unsupervised_data(skip_existing_directories=skip_existing_directories)


if __name__ == '__main__':
    main()
