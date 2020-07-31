Experiments
===========

This module contains all experiments.
Each experiment is an executable Python module
with a set of CLI switches.
Refer to the main README to see how experiments are run.

There are two types of experiments, search experiments and fit experiments.  

The search experiments are experiments, which conduct a hyperparameter search
via Bayesian optimization.
Each search experiment has a corresponding fit experiment
which hardcodes the best hyperparameters found during search.  

The fit experiments are trained on the full data using hardcoded parameters
found during search.
They also perform prediction on the test set after training.

The fit experiments are

- `baseline_dummy`: General dummy baselines to assess the data distribution
- `baseline_tiramisu`: Tiramisu model serving as a baseline
- `baseline_unet`: U-Net model serving as a baseline
- `fastfcn_contrastive`: FastFCN model with our proposed contrastive context regularization
- `fastfcn_modified_se_loss`: FastFCN model with a modified SE-loss
- `fastfcn_no_context`: FastFCN model with omitted context encoding module
- `fastfcn_no_se_loss`: FastFCN model with SE-loss weight set to 0

Since each search experiment corresponds to exactly one fit experiment
their description is omitted.
