Scripts
=======

This module contains various scripts.
Assuming the correct Python path they should be called as

    python -m road_segmentation.scripts.script_name

where `script_name` refers to the name of the script.

The following scripts currently exist

| Script                        | Description                                                                     | Usage                                                            |
|-------------------------------|---------------------------------------------------------------------------------|------------------------------------------------------------------|
| build_unsupervised_dataset.py | Converts raw NAIP tif mosaic images into patches usable for contrastive models. | `python -m road_segmentation.scripts.build_unsupervised_dataset` |
