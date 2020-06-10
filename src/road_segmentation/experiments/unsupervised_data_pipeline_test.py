import numpy as np
import argparse

import typing

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'Unsupervised Data Pipeline Test'
EXPERIMENT_TAG = 'unsupervised_data_pipeline_test'


class UnsupervisedDataPipelineExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
        parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'batch_size': args.batch_size,
            'dropout_rate': args.dropout_rate,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs
        }

    def fit(self) -> typing.Any:
        batch_size = self.parameters['batch_size']

        self.log.info('Loading training and validation data')
        try:
            image_paths = rs.data.cil.unsupervised_preprocessed_data_paths(self.data_directory)
            # images = rs.data.cil.load_images(image_paths)
            # validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            # self.log.debug(
            #    'Loaded %d training and %d validation samples',
            #    training_images.shape[0],
            #    validation_images.shape[0]
            # )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            exit()
            return

        print("!!! Not fully implemented !!!")
        print("!!! Exit !!!")
        exit()

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        raise NotImplementedError()


def main():
    UnsupervisedDataPipelineExperiment().run()


if __name__ == '__main__':
    main()
