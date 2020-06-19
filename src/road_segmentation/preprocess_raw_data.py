import gc
import os
import threading
import time
import typing
import warnings

from PIL import Image

import road_segmentation as rs

"""
This script process the unsupervised data.
First it loads the .tif files and extract patches, which are then stored as .png files
in a separate directory for each city and tile.

Each city is processed in a new thread
"""


def _process_city(tile_paths_per_city: typing.Dict[str, typing.List[str]],
                  city: str,
                  base_output_dir: str,
                  target_height: int,
                  target_width: int,
                  skip_existing_directories: bool):
    """
    Processes a city, i.e. extract patches for all tiles of a city
    """
    start_time = time.time()
    print("Processing {}... (Takes a few minutes)".format(city))
    total_number_of_patches = 0
    for i, tile_path in enumerate(tile_paths_per_city[city]):
        print("Tile {} of {} for {}".format(i + 1, len(tile_paths_per_city[city]), city))
        tile_name = tile_path.split("/")[-1][:-4]
        output_dir = os.path.join(base_output_dir, city, tile_name)
        if skip_existing_directories and os.path.exists(output_dir):
            print(f"Skip: {tile_name}")
            continue
        else:
            os.makedirs(output_dir)
        image = rs.data.cil.load_image(tile_path)
        image = image[:, :, :3]
        patches = rs.data.unsupervised.extract_patches_from_image(image, target_height, target_width)
        rs.data.unsupervised.save_images_to_png(patches, output_dir)

        total_number_of_patches += len(patches)
        if i % 10 == 0:
            gc.collect()
    end_time = time.time() - start_time
    print(f"Number of patches for {city}: {total_number_of_patches}\n" +
          f"Processing took {end_time} seconds")


def preprocess_unsupervised_data(target_height: int,
                                 target_width: int,
                                 data_dir: str = None,
                                 skip_existing_directories: bool = False):
    """
    Main method to run unsupervised data preprocessing.
    Extracts patches for each city and each tile into a separate directory.
    Each city is processed in a new thread
    Args:
        data_dir: in case data directory is different from default
        target_height: height of patches
        target_width: width of patches
        skip_existing_directories: if true then only tiles for which no output directory exists are processed
    """
    warnings.simplefilter('ignore', Image.DecompressionBombWarning)
    if data_dir is None:
        data_dir = rs.util.DEFAULT_DATA_DIR

    tile_paths_per_city = rs.data.unsupervised.raw_data_paths(data_dir)
    base_output_dir = os.path.join(data_dir, 'processed', "unsupervised")

    for city in rs.data.unsupervised.CITIES:
        threading.Thread(target=_process_city,
                         args=(tile_paths_per_city,
                               city,
                               base_output_dir,
                               target_height,
                               target_width,
                               skip_existing_directories)).start()


def main():
    target_image_width = 588
    target_image_height = 588
    skip_existing_directories = False  # if true then only tiles for which no output directory exists are processed
    preprocess_unsupervised_data(target_height=target_image_height,
                                 target_width=target_image_width,
                                 skip_existing_directories=skip_existing_directories)


if __name__ == '__main__':
    main()
