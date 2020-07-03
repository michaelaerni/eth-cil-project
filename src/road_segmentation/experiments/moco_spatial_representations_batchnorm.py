import argparse
import logging
import typing

import numpy as np
import tensorflow as tf

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'MoCo Spatial Representations Only - Using Batch Normalization'
EXPERIMENT_TAG = 'moco_spatial_representations_batchnorm'


def main():
    MoCoSpatialRepresentationsExperimentBatchNorm().run()


class MoCoSpatialRepresentationsExperimentBatchNorm(
    rs.experiments.moco_spatial_representations.MoCoSpatialRepresentationsExperiment
):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        # FIXME: [v1] The original does shuffling batch norm across GPUs to avoid issues stemming from
        #  leaking statistics via the normalization.
        #  We need a solution which works on a single GPU.
        #  arXiv:1905.09272 [cs.CV] does layer norm instead which is more suitable to our single-GPU case.
        #  This seems to fit similarly fast but we need to evaluate the effects on downstream performance.
        #  Furthermore, layer normalization requires much more memory than batch norm and thus reduces batch size etc.
        # TODO: [v1] try out (emulated) shuffling batch norm as well.
        #  This could be achieved by shuffling, splitting the batch and parallel batch norm layers.
        #  However, that might also be memory- and performance-inefficient.
        normalization_builder = rs.util.BatchNormalizationBuilder()

        kwargs = {
            'kernel_regularizer': tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
            'kernel_initializer': self.parameters['kernel_initializer'],
            'normalization_builder': normalization_builder
        }

        # TODO: Just as a general reminder, we need to implement the improved ResNet version!

        if name == 'ResNet50':
            return rs.models.resnet.ResNet50Backbone(**kwargs)
        if name == 'ResNet101':
            return rs.models.resnet.ResNet101Backbone(**kwargs)

        raise AssertionError(f'Unexpected backbone name "{name}"')


if __name__ == '__main__':
    main()
