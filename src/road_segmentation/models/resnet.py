import typing

import tensorflow as tf

"""
Implementation of individual blocks and full ResNet backbones according to (https://arxiv.org/abs/1512.03385).
"""


class ResNetBackbone(tf.keras.Model):
    """
    TODO: All documentation
    """

    _INITIAL_FILTERS = 64
    _KERNEL_INITIALIZER = 'he_normal'

    def __init__(
            self,
            blocks: typing.Iterable[int],
            weight_decay: float = 1e-4
    ):
        super(ResNetBackbone, self).__init__()

        # Layer 1: Initial convolution
        # TODO: The FastFCN implementation uses a 'deep_base' layer in which the 7x7 convolution
        #  is replaced by three consecutive 3x3 convolutions.
        #  This might be better in terms of segmentation performance
        #  but takes significant amounts of memory which we might not be able to afford.
        # Bias in the convolution layer is omitted since the batch normalization adds a bias term itself
        self.conv_in = tf.keras.layers.Conv2D(
            filters=self._INITIAL_FILTERS,
            kernel_size=7,
            strides=(2, 2),
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=self._KERNEL_INITIALIZER,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_in = tf.keras.layers.BatchNormalization()
        self.activation_in = tf.keras.layers.ReLU()

        # Layers 2 and later, starts with a max-pool layer
        self.pool_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.residual_layers = [
            ResNetLayer(
                current_blocks,
                self._INITIAL_FILTERS * (2 ** idx),
                downsample=(idx > 0),  # No downsampling on first ResNet block due to the initial max pooling
                kernel_initializer=self._KERNEL_INITIALIZER,
                weight_decay=weight_decay
            )
            for idx, current_blocks in enumerate(blocks)
        ]

    def call(self, inputs, training=None, mask=None):
        # Store the features of each block for output
        block_features = []

        # Initial convolution
        with tf.keras.backend.name_scope('conv1'):
            initial_features = self.conv_in(inputs)
            initial_features = self.batch_norm_in(initial_features)
            initial_features = self.activation_in(initial_features)
        block_features.append(initial_features)

        # Max-pooling is done on layer 2, here explicitly to simplify the loop
        with tf.keras.backend.name_scope('conv2_pre'):
            features = self.pool_in(initial_features)

        # Layers 2 and later
        for layer_number, current_layer in enumerate(self.residual_layers, start=2):
            with tf.keras.backend.name_scope(f'conv{layer_number}'):
                features = current_layer(features)
            block_features.append(features)

        return block_features


class ResNet50Backbone(ResNetBackbone):
    """
    Convenience class to instantiate a ResNet-50 backbone with the correct number of blocks per layer.
    """

    def __init__(self, weight_decay: float = 1e-4):
        super(ResNet50Backbone, self).__init__(
            blocks=[3, 4, 6, 3],
            weight_decay=weight_decay
        )


class ResNet101Backbone(ResNetBackbone):
    """
    Convenience class to instantiate a ResNet-101 backbone with the correct number of blocks per layer.
    """

    def __init__(self, weight_decay: float = 1e-4):
        super(ResNet101Backbone, self).__init__(
            blocks=[3, 4, 23, 3]
        )


class ResNetLayer(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            blocks: int,
            initial_features: int,
            downsample: bool,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            weight_decay: float,
            **kwargs
    ):
        super(ResNetLayer, self).__init__(**kwargs)

        self.blocks = [
            BottleneckBlock(
                initial_features,
                downsample=(downsample and idx == 0),  # Downsampling always on first block (if desired)
                projection_shortcut=(idx == 0),  # Projection shortcut always on first block
                kernel_initializer=kernel_initializer,
                weight_decay=weight_decay
            )
            for idx in range(blocks)
        ]

    def call(self, inputs, **kwargs):
        features = inputs
        for current_block in self.blocks:
            features = current_block(features)

        return features


class BottleneckBlock(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            filters_in: int,
            downsample: bool,
            projection_shortcut: bool,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            weight_decay: float,
            **kwargs
    ):
        super(BottleneckBlock, self).__init__(**kwargs)

        # Bias in convolution layers is omitted since the batch normalizations add a bias term themselves

        # Number of filters grows by factor of 4
        filters_out = 4 * filters_in

        # Initial 1x1 convolution
        strides_in = (2, 2) if downsample else (1, 1)
        self.conv_in = tf.keras.layers.Conv2D(
            filters=filters_in,
            kernel_size=1,
            strides=strides_in,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_in = tf.keras.layers.BatchNormalization()
        self.activation_in = tf.keras.layers.ReLU()

        # Middle 3x3 convolution
        self.conv_middle = tf.keras.layers.Conv2D(
            filters=filters_in,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_middle = tf.keras.layers.BatchNormalization()
        self.activation_middle = tf.keras.layers.ReLU()

        # Output 1x1 convolution without activation (is done externally) and 4x as many filters
        self.conv_out = tf.keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_out = tf.keras.layers.BatchNormalization()
        self.activation_out = tf.keras.layers.ReLU()

        # Projection shortcut if required (i.e. when downsampling or changing the number of features)
        self.conv_residual = None
        self.batch_norm_residual = None
        if downsample or projection_shortcut:
            self.conv_residual = tf.keras.layers.Conv2D(
                filters=filters_out,
                kernel_size=1,
                strides=strides_in,
                padding='same',
                activation=None,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            )
            self.batch_norm_residual = tf.keras.layers.BatchNormalization()
            # No activation here, is done on all features on output

    def call(self, inputs, **kwargs):
        # Perform projection shortcut if necessary
        residuals = inputs
        if self.conv_residual is not None:
            assert self.batch_norm_residual is not None
            residuals = self.conv_residual(residuals)
            residuals = self.batch_norm_residual(residuals)

        # Differentiate actual block from residual handling
        with tf.keras.backend.name_scope('block'):
            features = self.conv_in(inputs)
            features = self.batch_norm_in(features)
            features1 = self.activation_in(features)

            features = self.conv_middle(features1)
            features = self.batch_norm_middle(features)
            features2 = self.activation_middle(features)

            features = self.conv_out(features2)
            block_output = self.batch_norm_out(features)

        # Add residuals
        pre_activation = block_output + residuals

        # Perform final activation
        output = self.activation_out(pre_activation)

        return output
