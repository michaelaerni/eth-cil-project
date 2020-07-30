CIL Road Segmentation 2020
==========================

Setup
-----
Through out this readme we refer to the main directory of this repository as <repo>

To easily use that project conda is required.
It is also possible to set up the project without conda,
however we highly recommend to use conda and follow this setup guide, to make it reproducible.

[Here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) one can get the latest installation instruction for conda.

After successfuly installing conda create the environment,
which includes all required packages in the correct version,
by executing the following command in the main directory of this repository:   
$ conda env update --file environment.yml --prefix .env/

Afterwards we can activate the environment by executing the command:  
$ conda activate /home/nic/PycharmProjects/ETH/eth-cil-project/.env


### Supervised data setup
The supervised data can be downloaded as a .zip from [kaggle](https://www.kaggle.com/c/cil-road-segmentation-2020).
After downlaoding place it under data/raw/cil-road-segmentation-2020/ and extract it there.


### Unsupervised data setup
TODO

Project Structure
-----------------
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

The experiment and script documentation is found in a
README in the corresponding module directories.