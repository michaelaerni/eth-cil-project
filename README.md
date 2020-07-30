CIL Road Segmentation 2020
==========================
This README generally assumes all paths to be relative to
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
TODO
The project structure is inspired by the [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science/) structre.

The general structure is as follows:
- /data contains all data (unversioned)
- /logs contains logs (e.g. model snapshots, submissions, etc), one directory per experiment run (unversioned)
- /src contains all code, except notebooks
- /notebooks contains all jupyter notebooks

The root module road_segmentation is located under src and its structure is as follows:
- road_segmentation.models contains all models
- road_segmentation.data contains data-handling code
- road_segmentation.experiments contains executable scripts to search hyperparameters, train and evaluate models
- road_segmentation.scripts contains executable scripts for other things, e.g. unsupervised data set generation
- road_segmentation.util contains helper functionality, e.g. fixing seeds
- road_segmentation.framework contains experimental framework code


Running Scripts and Experiments
--------------------------------
TODO
The experiment and script documentation is found in a
README in the corresponding module directories.
The experiment and script documentation is found in a
README in the corresponding module directories.


Unsupervised Data Set
---------------------
TODO: Document unsupervised data set source and creation

TODO: Include CSV documenting the raw unsupervised data
