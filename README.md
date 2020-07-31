CIL Road Segmentation 2020
==========================
This README generally assumes a *nix operating system
and further all paths to be relative to
the project root directory (for commands and directories)
unless otherwise mentioned.
The project root directory is the directory containing this README file.


Setup
-----
### Dependencies
We use [Conda](https://docs.conda.io) to manage the project environment.
This abstracts away all Python and system dependencies (e.g. CUDA).
Thus, all code should be easily runnable without
having to install any system-wide dependencies except for Conda.
We recommend and tested
the [Miniconda](https://docs.conda.io/en/latest/miniconda.html) distribution.

To create a new environment for the project run

    conda env create --file environment.yml --prefix .env/

in the project root directory.
This only needs to be done once and
creates a new environment in `.env/`.

The environment must to be activated to be used.
This is done by issuing

    conda activate .env/

in the project root directory.
To deactivate the environment again run

    conda deactivate

anywhere.

Any changes to dependencies should happen in `environemnt.yml`.
After changing dependencies the environment must be updated by issuing

    conda env create --file environment.yml --prefix .env/

from the project root directory.

### Supervised Data Set
The supervised data can be downloaded from
the [Kaggle project page](https://www.kaggle.com/c/cil-road-segmentation-2020).
It must be extracted into
`data/raw/cil-road-segmentation-2020/`
without any additional subdirectories.

### Unsupervised Data Set
See *Unsupervised Data Set* regarding how to build
the unsupervised data set from scratch.

The processed unsupervised data set is over 60GB in size and
we pay for the traffic ourselves.
Thus, a link can only be made available upon request.
The processed version of the data set
will be retained until *Wednesday, 9th of September 2020*.
After that date it must always be recreated from scratch.

If the processed unsupervised data set is downloaded
it must be extracted directly into
`data/processed/unsupervised/`
without any additional subdirectories
(resulting in paths such as `data/processed/unsupervised/Boston/`).


Project Structure
-----------------
The project structure is inspired by
[cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science/)
but heavily tailored for our purpose.

### Directory Structure
The general structure relative to the root directory is as follows:
- `data/` is a placeholder directory for data sets.
    The repository contains only the directory structure,
    not the actual data sets.
- `log/` is a placeholder directory for experimental outputs.
    Each experiment run creates a unique subdirectory
    containing parameters, logs, and outputs.
    Experiment runs are excluded from version control.
- `src/` contains all normal Python code of the project.
- `notebooks/` contains all Jupyter notebooks.

### Code Structure
The code is organised in a proper Python module structure.
The root module is `road_segmentation`.
Modules recursively include each other such that all sub-modules
can be referenced by simply importing `road_segmentation`.

We use the pattern to import `road_segmentation` as `rs`
and then reference sub-modules directly in our code.
As an example,
we reference `rs.data.cil` directly
instead of importing `road_segmentation.data.cil` manually.

The module structure is as follows:
- `road_segmentation`: Root module
    - `road_segmentation.framework`: Shared code of the experimental framework
    - `road_segmentation.metrics`: Custom Keras metrics
    - `road_segmentation.util`: Various utility methods
    - `road_segmentation.data`: Data loading and processing
        - `road_segmentation.data.cil`:
            Functionality related to the provided labelled data set
        - `road_segmentation.data.unsupervised`:
            Functionality related to our new unlabelled data set
        - `road_segmentation.data.image`:
            Various image augmentation and processing methods
    - `road_segmentation.experiments`: Root module for experiments
        - `road_segmentation.experiments.search`:
            Root module of experiments for hyperparameter search
        - `road_segmentation.experiments.fit`:
            Root module of experiments for fitting a classifier using a single fixed set of hyperparameters
        - `road_segmentation.experiments.unused`:
            Root module of old irrelevant experiments
            created while exploring possibilities,
            to be ignored
    - `road_segmentation.models`: Root module for actual models used in experiments
    - `road_segmentation.scripts`: Root module for various scripts

The actual experiments are documented in a README within the
`road_segmentation.experiments` module.
The actual scripts are documented in a README within the
`road_segmentation.scripts` module.

Our code follows the [PEP 8](https://www.python.org/dev/peps/pep-0008/)
guidelines with minor modifications to follow modern best-practices
regarding clean code.
Type annotations are used and respected wherever possible.
Comments and docstrings follow the
[Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

### Dependencies
A detailed list of all dependencies (up to minor version)
including system libraries managed via Conda can be found
in `environment.yml`.

The key dependencies are
*Python 3.7* as the programming language,
*TensorFlow 2.1* with *Keras* for deep learning model fitting,
and the *AX Platform 0.1* for hyperparameter search.
Note that the *AX Platform* transitively depends on *PyTorch*,
explaining the presence of two deep learning frameworks.


Running Scripts and Experiments
--------------------------------
Since the whole code is an importable Python module,
running a script or experiment simply corresponds
to executing a Python module.

Let `$PROJECT_ROOT` denote the absolute path to the
project root directory.
The preferred way to run any experiment or script
of this project is to prepend the `src/` directory
to the local `$PYTHONPATH` environment variable via

    PYTHONPATH="$PROJECT_ROOT/src/:$PYTHONPATH"

and then running the target module, e.g.

    python -m road_segmentation.scripts.build_unsupervised_dataset

It is also possible to leave the environment variable
generally intact and only change it during invocation
by prepending it to the actual command, e.g.

    PYTHONPATH="$PROJECT_ROOT/src/:$PYTHONPATH" python -m road_segmentation.scripts.build_unsupervised_dataset

All experiments and scripts provide a command line interface
with built-in help functionality explaining the arguments.


Unsupervised Data Set
---------------------
We have used USDA's NAIP imagery for our unsupervised data set, which provides aerial imagery of the USA. It can be acquired through USGS's [EarthExplorer](https://earthexplorer.usgs.gov/) interface. An account has to be made to be able to download the imagery.

In the project we have acquired data over 5 cities with varying size and layout. Each dataset's metadata is provided in a CSV file in the `data/raw/unsupervised/` folder. At the writing of this document there is a new method to extract the imagery with the CSV file automatically, but due to errors on the providers side we were not able to test this feature. A working solution is to use the EarthExplorer site manually.

* Under the Search Criteria tab there are no criteria to be made.
* Under the Data Sets tab select `Aerial Imagery > NAIP`.
* Under the Additional Criteria tab the Entity ID's of each imagery from the CSV file has to be manually entered. An example Entity ID from the Milwaukee set is `M_4208806_NE_16_060_20180917`. Multiple Entity IDs can be used in a single search and all valid imagery will be listed.
* Under the Results page one has to manually proceed. For each entity click the Download options button (5th) and select Download. At the time of data acquisition it was possible to download 9 imagery in parallel with this method.

Recently a new method has been added, which allows the use of CSV files to download through the Bulk Download Application. The application was unavailable at the writing of this document, but the issues might have been fixed since then. This method requires the download of additional software.
* From the EarthExplorer site navigate to the Item Basket without adding any item. 
* Under Scene List > File Upload Options select `Metadata CSV Export`.
* Use `NAIP` as Data Set, select an appropriate File List Name, and upload the CSV file provided.
* Select Upload File in the upper left corner of the table.
* On the bottom of the page Select Order Type (Bulk) and the Selected List (newly uploaded File List Name).
* Follow the [Bulk Download Tutorial](https://lta.cr.usgs.gov/EEHelp/ee_help).
Note: sometimes it takes longer to receive the confirmation for the order.

The raw unsupervised data set must be downloaded/moved into
`data/raw/unsupervised/[cityname]`
without any additional subdirectories for each city.
(resulting in paths such as `data/raw/unsupervised/Boston/`).

Each folder will contain several larger imagery, that will cover the city in a mosaic. Therefore we call an imagery a tile in the next section.

### Processing

The unsupervised data set can be processed with the build_unsupervised_dataset script.

From each tile (imagery) in the raw data set we extract as many patches as possible without overlapping in a single tile. The extracted patches form an image-aligned, image-centered grid. This means that the edges of the tiles will be cropped depending on the patch size. On each side at most PATCH_SIZE/2 sized area will be cropped. Due to the nature of the raw data set, the tiles are overlapping, which results in overlaps between patches from different tiles, but also covers the cropped parts.
