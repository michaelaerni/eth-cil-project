import argparse
import logging
import typing
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import road_segmentation as rs

EXPERIMENT_DESCRIPTION = 'FastFCN Downstream Experiment'
EXPERIMENT_TAG = 'fastfcn_downstream'


def main():
    FastFCNDownstreamExperiment().run()


class FastFCNDownstreamExperiment(rs.framework.Experiment):

    @property
    def tag(self) -> str:
        return EXPERIMENT_TAG

    @property
    def description(self) -> str:
        return EXPERIMENT_DESCRIPTION

    def create_argument_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Defaults are roughly based on ADE20k experiments of the original paper
        parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')  # FIXME: Was 16 originally
        parser.add_argument('--learning-rate', type=float, default=1e-2, help='Initial learning rate')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for convolution weights')
        parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
        parser.add_argument('--segmentation-loss-weight', type=float, default=1.0, help='Weight of segmentation loss')
        parser.add_argument('--encoder-loss-weight', type=float, default=0.2, help='Weight of modified SE loss')
        parser.add_argument(
            '--backbone',
            type=str,
            default='ResNet50',
            choices=('ResNet50', 'ResNet101'),
            help='Backbone model type to use'
        )
        parser.add_argument(
            '--backbone-checkpoint',
            type=str,
            help='.h5 File with saved model weights.'
        )

        return parser

    def build_parameter_dict(self, args: argparse.Namespace) -> typing.Dict[str, typing.Any]:
        return {
            'jpu_features': 512,  # FIXME: We could decrease those since we have less classes.
            'backbone': args.backbone,
            'weight_decay': args.weight_decay,
            'segmentation_loss_weight': args.segmentation_loss_weight,
            'encoder_loss_weight': args.encoder_loss_weight,
            'head_dropout': 0.1,
            'output_upsampling': 'nearest',
            'kernel_initializer': 'he_normal',  # FIXME: This might not necessarily be the best choice
            'dense_initializer': 'he_uniform',  # Only for the dense weights in the Encoder head
            'batch_size': args.batch_size,
            'initial_learning_rate': args.learning_rate,
            'end_learning_rate': 1e-8,  # FIXME: The original authors decay to zero but small non-zero might be better
            'learning_rate_decay': 0.9,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'prefetch_buffer_size': 16,
            'augmentation_max_relative_scaling': 0.04,  # Scaling +- one output feature, result in [384, 416]
            'augmentation_interpolation': 'bilinear',
            'augmentation_blur_probability': 0.5,
            'augmentation_blur_size': 5,  # 5x5 Gaussian filter for blurring
            'training_image_size': (384, 384),
            'backbone_checkpoint': args.backbone_checkpoint
        }

    def fit(self) -> typing.Any:
        self.log.info('Loading training and validation data')
        try:
            trainig_paths, validation_paths = rs.data.cil.train_validation_sample_paths(self.data_directory)
            training_images, training_masks = rs.data.cil.load_images(trainig_paths)
            validation_images, validation_masks = rs.data.cil.load_images(validation_paths)
            self.log.debug(
                'Loaded %d training and %d validation samples',
                training_images.shape[0],
                validation_images.shape[0]
            )
        except (OSError, ValueError):
            self.log.exception('Unable to load data')
            return

        training_dataset = tf.data.Dataset.from_tensor_slices((training_images, training_masks))
        training_dataset = training_dataset.shuffle(buffer_size=training_images.shape[0])
        training_dataset = training_dataset.map(lambda image, mask: self._augment_sample(image, mask))
        training_dataset = training_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        training_dataset = training_dataset.batch(self.parameters['batch_size'])
        training_dataset = training_dataset.prefetch(buffer_size=self.parameters['prefetch_buffer_size'])
        self.log.debug('Training data specification: %s', training_dataset.element_spec)

        # Validation images can be directly converted to the model colour space
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (rs.data.image.rgb_to_cielab(validation_images), validation_masks)
        )
        validation_dataset = validation_dataset.map(lambda image, mask: self._calculate_se_loss_target(image, mask))
        validation_dataset = validation_dataset.batch(1)

        # Build model
        self.log.info('Building model')
        backbone = self._construct_backbone(self.parameters['backbone'])

        model = rs.models.fastfcn_moco.FastFCNMoCo(
            backbone,
            kernel_initializer=self.parameters['kernel_initializer'],
            dense_initializer=self.parameters['dense_initializer'],
            output_upsampling=self.parameters['output_upsampling'],
            dropout_rate=self.parameters['head_dropout'],
            kernel_regularizer=tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay'])
        )

        model.build(training_dataset.element_spec[0].shape)

        if not os.path.exists(self.parameters['backbone_checkpoint']):
            raise FileNotFoundError(
                'Checkpoint file "{}" does not exist'.format(self.parameters['backbone_checkpoint']))
        backbone.load_weights(self.parameters['backbone_checkpoint'])

        # Log model structure if debug logging is enabled
        if self.log.isEnabledFor(logging.DEBUG):
            model.summary(
                line_length=120,
                print_fn=lambda s: self.log.debug(s)
            )

        metrics = {
            'output_1': self.keras.default_metrics(threshold=0.0)
        }

        # TODO: Check whether the binary cross-entropy loss behaves correctly
        losses = {
            'output_1': tf.keras.losses.BinaryCrossentropy(from_logits=True),  # Segmentation loss
            'output_2': tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Modified SE-loss
        }
        loss_weights = {
            'output_1': self.parameters['segmentation_loss_weight'],
            'output_2': self.parameters['encoder_loss_weight']
        }

        steps_per_epoch = np.ceil(training_images.shape[0] / self.parameters['batch_size'])
        self.log.debug('Calculated steps per epoch: %d', steps_per_epoch)
        # TODO: This performs weight decay on an optimizer level, not on a case-by-case basis.
        #  There's a difference!
        #  Global weight decay might be dangerous if we also have the Encoder head (with the parameters there)
        #  but it could also be an important ingredient for success...
        optimizer = self._build_optimizer(steps_per_epoch)
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )

        callbacks = [
            self.keras.tensorboard_callback(),
            self.keras.periodic_checkpoint_callback(),
            self.keras.best_checkpoint_callback(metric='val_output_1_binary_mean_f_score'),
            self.keras.log_predictions(
                validation_images=rs.data.image.rgb_to_cielab(validation_images),
                display_images=validation_images,
                prediction_idx=0
            )
        ]

        # Fit model
        model.fit(
            training_dataset,
            epochs=self.parameters['epochs'],
            validation_data=validation_dataset,
            callbacks=callbacks
        )

        return model

    def predict(self, classifier: typing.Any, images: typing.Dict[int, np.ndarray]) -> typing.Dict[int, np.ndarray]:
        result = dict()

        for sample_id, image in images.items():
            self.log.debug('Predicting sample %d', sample_id)
            image = np.expand_dims(image, axis=0)

            # Convert to model colour space
            image = rs.data.image.rgb_to_cielab(image)

            (raw_prediction,), _ = classifier.predict(image)
            prediction = np.where(raw_prediction >= 0, 1, 0)

            result[sample_id] = prediction

        return result

    def _augment_sample(self, image: tf.Tensor, mask: tf.Tensor) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        # Random Gaussian blurring
        do_blur = tf.random.uniform(shape=[], dtype=tf.float32) < self.parameters['augmentation_blur_probability']
        blurred_image = tf.cond(do_blur, lambda: self._augment_blur(image), lambda: image)
        blurred_image.set_shape(image.shape)  # Must set shape manually since it cannot be inferred from tf.cond

        # Random scaling
        scaling_factor = tf.random.uniform(
            shape=[],
            minval=1.0 - self.parameters['augmentation_max_relative_scaling'],
            maxval=1.0 + self.parameters['augmentation_max_relative_scaling']
        )
        input_height, input_width, _ = tf.unstack(tf.cast(tf.shape(blurred_image), tf.float32))
        scaled_size = tf.cast(
            tf.round((input_height * scaling_factor, input_width * scaling_factor)),
            tf.int32
        )
        scaled_image = tf.image.resize(blurred_image, scaled_size, method=self.parameters['augmentation_interpolation'])
        scaled_mask = tf.image.resize(mask, scaled_size, method='nearest')

        # Combine image and mask to ensure same transformations are applied
        concatenated_sample = tf.concat((scaled_image, scaled_mask), axis=-1)

        # Random flip and rotation, this covers all possible permutations which do not require interpolation
        flipped_sample = tf.image.random_flip_left_right(concatenated_sample)
        num_rotations = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        rotated_sample = tf.image.rot90(flipped_sample, num_rotations)

        # Random crop
        crop_size = self.parameters['training_image_size'] + (4,)  # 3 colour channels + 1 mask channel
        cropped_sample = tf.image.random_crop(rotated_sample, crop_size)

        # Split combined image and mask again
        output_image = cropped_sample[:, :, :3]
        output_mask = cropped_sample[:, :, 3:]

        # Convert mask to labels in {0, 1} but keep as floats
        output_mask = tf.round(output_mask)

        # Convert image to CIE Lab
        # This has to be done after the other transformations since some assume RGB inputs
        output_image_lab = rs.data.image.map_colorspace(output_image)

        # FIXME: It would make sense to apply colour shifts but the original paper does not

        return output_image_lab, output_mask

    def _augment_blur(self, image: tf.Tensor) -> tf.Tensor:
        # Pick standard deviation randomly in [0.5, 1)
        sigma = tf.random.uniform(shape=[], minval=0.5, maxval=1.0, dtype=tf.float32)
        sigma_squared = tf.square(sigma)

        # FIXME: This would be significantly faster if applied as two 1D convolutions instead of a 2D one

        # Calculate Gaussian filter kernel
        kernel_size = self.parameters['augmentation_blur_size']
        half_kernel_size = kernel_size // 2
        grid_y_squared, grid_x_squared = np.square(
            np.mgrid[-half_kernel_size:half_kernel_size + 1, -half_kernel_size:half_kernel_size + 1]
        )
        coordinates = grid_y_squared + grid_x_squared
        kernel = 1.0 / (2.0 * np.pi * sigma_squared) * tf.exp(
            - coordinates / (2.0 * sigma_squared)
        )
        kernel = tf.reshape(kernel, (kernel_size, kernel_size, 1, 1))
        kernel = tf.repeat(kernel, 3, axis=2)
        # => Kernel shape is [kernel_size, kernel_size, 3, 1]

        # Pad image using reflection padding (not available in depthwise_conv2d)
        padded_image = tf.pad(
            image,
            paddings=((half_kernel_size, half_kernel_size), (half_kernel_size, half_kernel_size), (0, 0)),
            mode='REFLECT'
        )
        padded_image = tf.expand_dims(padded_image, axis=0)

        # Finally apply Gaussian filter
        blurred_image = tf.nn.depthwise_conv2d(padded_image, kernel, strides=(1, 1, 1, 1), padding='VALID')

        # Result might have values outside the normalized range, clip those
        output = tf.clip_by_value(blurred_image[0], 0.0, 1.0)
        return output

    # noinspection PyMethodMayBeStatic
    def _calculate_se_loss_target(
            self,
            image: tf.Tensor,
            mask: tf.Tensor
    ) -> typing.Tuple[tf.Tensor, typing.Tuple[tf.Tensor, tf.Tensor]]:
        # The target to predict is the logit of the proportion of foreground pixels (i.e. empirical prior)
        foreground_prior = tf.reduce_mean(mask)

        return image, (mask, foreground_prior)

    def _construct_backbone(self, name: str) -> tf.keras.Model:
        resnet = None
        if name == 'ResNet50':
            resnet = rs.models.resnet.ResNet50Backbone(
                kernel_initializer=self.parameters['kernel_initializer']
            )
        elif name == 'ResNet101':
            resnet = rs.models.resnet.ResNet101Backbone(
                kernel_initializer=self.parameters['kernel_initializer']
            )
        else:
            raise AssertionError(f'Unexpected backbone name "{name}"')

        moco_backbone = rs.models.fastfcn_moco.FastFCNMoCoBackbone(
            resnet,
            kernel_initializer=self.parameters['kernel_initializer'],
            kernel_regularizer=tf.keras.regularizers.L1L2(l2=self.parameters['weight_decay']),
            jpu_features=self.parameters['jpu_features']
        )
        return moco_backbone

    def _build_optimizer(self, steps_per_epoch: int) -> tfa.optimizers.SGDW:
        learning_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.parameters['initial_learning_rate'],
            decay_steps=self.parameters['epochs'] * steps_per_epoch,
            end_learning_rate=self.parameters['end_learning_rate'],
            power=self.parameters['learning_rate_decay']
        )

        # Determine the weight decay schedule proportional to the learning rate decay schedule
        weight_decay_factor = self.parameters['weight_decay'] / self.parameters['initial_learning_rate']

        # This has to be done that way since weight_decay needs to access the optimizer lazily, hence the lambda
        optimizer = tfa.optimizers.SGDW(
            weight_decay=lambda: weight_decay_factor * learning_rate_scheduler(optimizer.iterations),
            learning_rate=learning_rate_scheduler,
            momentum=self.parameters['momentum']
        )

        return optimizer


if __name__ == '__main__':
    main()
