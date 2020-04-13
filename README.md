CIL Road Segmentation 2020
==========================

Setup
-----
TODO

TODO: conda setup

TODO: conda update: conda env update --file environment.yml --prefix .env/

TODO: Data: Download zip from kaggle, place in data/raw/cil-road-segmentation-2020/ and extract there


Project Structure
-----------------
TODO: Inspired by https://github.com/drivendata/cookiecutter-data-science/

TODO: General structure
- /data contains all data (unversioned)
- /logs contains logs (e.g. model snapshots, submissions, etc), one directory per experiment run (unversioned)
- /src contains all code

TODO: Code module structure
- road_segmentation is the root module
- road_segmentation.models contains all models
- road_segmentation.data contains data-handling code
- road_segmentation.experiments contains executable scripts to train and evaluate models
- road_segmentation.util contains helper functionality, e.g. to evaluate models or boilerplate experiment code
- road_segmentation.framework contains experimental framework code
