Experiments
===========

This module contains all experiments.

We discriminate between two types of experiments, search experiments and fit experiments.  
The search experiments are experiments, which conduct a hyperparameter search via bayesian optimization.
Each search experiments has a corresponing fit experiment which hardcodes the best hyperparameters found during search.  
The fit experiments are trained on the full data and uses the hardcoded parameters found during search. 

The structure is as follows:
 - /search contains all search experiments
 - /fit contains all fit experiments
 - /unused contains old experiments which are not relevant anymore, but kept for reference
 
Baseline FastFCN without Context Encoding Module
----------------------------------
TODO: Document

Baseline FastFCN without SE-Loss
----------------------------------
TODO: Document

Baseline FastFCN with modified SE-Loss
----------------------------------
TODO: Document

Baseline FastFCN with MoCo Context
----------------------------------
TODO: Document


## Run the experiments

Assuming the correct Python path they should be called as

    python -m road_segmentation.experiments.experiment_type.experiment_name

where `experiment_type` refers to the type of the experiment and `experiment_name` refers to the name of the experiment.

For example to start the search experiment for the Baseline FastFCN with MoCo Context experiment use the following command:
`python -m road_segmentation.experiments.search.fastfcn_moco_context_search` 
