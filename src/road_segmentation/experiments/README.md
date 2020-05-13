Experiments
===========

Old Baseline (Logistic Regression)
----------------------------------
TODO: Document


Tiramisu
--------
Implementation of the three FCDenseNets, (FCDenseNet56, FCDenseNet67 and FCDenseNet103) introduced in this [paper](https://arxiv.org/abs/1611.09326).

The experiment is as truthful to the original paper as possible, and serves as a baseline to compare
other experiments to.
However, there are a few differences to the original:
 - Training batch size: We use a batch size of 1, while the original paper uses a batch size of 3 for CamVid and batch size 5 for Gatech.
   This however results in OOM issues for us.
 - Batch Normalisation: The paper explicitly uses current batch statistics at training, validation and test time, while we use whatever Keras does by default.
 - Output of last dense block: We implement it as described in the paper: in the up path the inputs to the dense blocks are not concatenated to their outputs.
   In the code accompanying the paper, this is however done for the very last dense block.
 - For early stopping we use the validation binary mean f score. The paper uses mIoU or mean accuracy.
