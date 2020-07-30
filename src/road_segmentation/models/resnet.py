import typing

import tensorflow as tf

import road_segmentation as rs

"""
Implementation of individual blocks and full ResNet backbones according to (https://arxiv.org/abs/1512.03385).
"""


class ResNetBackbone(tf.keras.Model):
    """
    ResNet-based segmentation backbone.

    This is essentially a ResNet without the fully connected layers.
    Calling this model yields a list containing the features of each "layer" in increasing stride.
    """

    _INITIAL_FILTERS = 64

    def __init__(
            self,
            blocks: typing.Iterable[int],
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_normal',
            normalization_builder: rs.util.NormalizationBuilder = None,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None
    ):
        """
        Create a new ResNet backbone.

        Args:
            blocks: Number of blocks per layer in increasing layer order.
                Thus, the number of entries in the list determines the number of layers and the output stride.
            kernel_initializer: Initializer for convolution kernels.
            normalization_builder: Normalization layer builder. Defaults to batch normalization.
            kernel_regularizer: Regularizer for convolution kernels.
        """

        super(ResNetBackbone, self).__init__()

        # Initialize defaults
        if normalization_builder is None:
            normalization_builder = rs.util.BatchNormalizationBuilder()

        # Layer 1: Initial convolution
        # Bias in the convolution layer is omitted since the normalization adds a bias term itself
        #
        # The FastFCN implementation uses a 'deep_base' layer in which the 7x7 convolution
        #  is replaced by three consecutive 3x3 convolutions.
        #  This might be better in terms of segmentation performance
        #  but takes significant amounts of memory which we might not be able to afford.
        self.conv_in = tf.keras.layers.Conv2D(
            filters=self._INITIAL_FILTERS,
            kernel_size=7,
            strides=(2, 2),
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.normalization_in = normalization_builder()
        self.activation_in = tf.keras.layers.ReLU()

        # Layers 2 and later, starts with a max-pool layer
        self.pool_in = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.residual_layers = [
            ResNetLayer(
                current_blocks,
                self._INITIAL_FILTERS * (2 ** idx),
                downsample=(idx > 0),  # No downsampling on first ResNet block due to the initial max pooling
                kernel_initializer=kernel_initializer,
                normalization_builder=normalization_builder,
                kernel_regularizer=kernel_regularizer
            )
            for idx, current_blocks in enumerate(blocks)
        ]

    def call(self, inputs, training=None, mask=None):
        """
        Call this backbone.

        Args:
            inputs: Batch of 3 channel images.
            training: Additional argument, unused.
            mask: Additional argument, unused.

        Returns:
            List containing the features of each "layer" in increasing stride.
        """

        # Store the features of each block for output
        block_features = []

        # Initial convolution
        with tf.keras.backend.name_scope('conv1'):
            initial_features = self.conv_in(inputs)
            initial_features = self.normalization_in(initial_features)
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

    def __init__(
            self,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_normal',
            normalization_builder: rs.util.NormalizationBuilder = None
    ):
        """
        Create a new ResNet-50 backbone.

        Args:
            kernel_regularizer: Regularizer for convolution kernels.
            kernel_initializer: Initializer for convolution kernels.
            normalization_builder: Normalization layer builder. Defaults to batch normalization.
        """

        super(ResNet50Backbone, self).__init__(
            blocks=[3, 4, 6, 3],
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            normalization_builder=normalization_builder
        )


class ResNet101Backbone(ResNetBackbone):
    """
    Convenience class to instantiate a ResNet-101 backbone with the correct number of blocks per layer.
    """

    def __init__(
            self,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_normal',
            normalization_builder: rs.util.NormalizationBuilder = None
    ):
        """
        Create a new ResNet-101 backbone.

        Args:
            kernel_regularizer: Regularizer for convolution kernels.
            kernel_initializer: Initializer for convolution kernels.
            normalization_builder: Normalization layer builder. Defaults to batch normalization.
        """

        super(ResNet101Backbone, self).__init__(
            blocks=[3, 4, 23, 3],
            kernel_regularizer=kernel_regularizer,
            kernel_initializer=kernel_initializer,
            normalization_builder=normalization_builder
        )


class ResNetLayer(tf.keras.layers.Layer):
    """
    Conceptual ResNet layer consisting of a series of blocks with residual connections.
    """

    def __init__(
            self,
            blocks: int,
            initial_features: int,
            downsample: bool,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            normalization_builder: rs.util.NormalizationBuilder,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Create a new ResNet layer.

        Args:
            blocks: Number of blocks making up this layer.
            initial_features: Number of initial features in each block.
            downsample: If True then the first block of this layer performs downsampling by a factor of 2x2.
            kernel_initializer: Initializer for convolution kernels.
            normalization_builder: Normalization layer builder.
            kernel_regularizer: Regularizer for convolution kernels.
            **kwargs: Additional arguments passed to `tf.keras.layers.Layer`.
        """

        super(ResNetLayer, self).__init__(**kwargs)

        self.blocks = [
            BottleneckBlock(
                initial_features,
                downsample=(downsample and idx == 0),  # Downsampling always on first block (if desired)
                projection_shortcut=(idx == 0),  # Projection shortcut always on first block
                kernel_initializer=kernel_initializer,
                normalization_builder=normalization_builder,
                kernel_regularizer=kernel_regularizer
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
    ResNet bottleneck block.
    """

    def __init__(
            self,
            filters_in: int,
            downsample: bool,
            projection_shortcut: bool,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            normalization_builder: rs.util.NormalizationBuilder,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Create a new ResNet bottleneck block.

        Args:
            filters_in: Number of input features. The number of output features will be 4x this value.
            downsample: If true the spatial resolution is reduced by a factor of 2x2.
            projection_shortcut: If True a projection shortcut is used. Else, an additive residual shortcut is used.
            kernel_initializer: Initializer for convolution kernels.
            normalization_builder: Normalization layer builder.
            kernel_regularizer: Regularizer for convolution kernels.
            **kwargs: Additional arguments passed to `tf.keras.layers.Layer`.
        """

        super(BottleneckBlock, self).__init__(**kwargs)

        # Bias in convolution layers is omitted since the normalizations add a bias term themselves

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
            kernel_regularizer=kernel_regularizer
        )
        self.normalization_in = normalization_builder()
        self.activation_in = tf.keras.layers.ReLU()

        # Middle 3x3 convolution
        self.conv_middle = tf.keras.layers.Conv2D(
            filters=filters_in,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.normalization_middle = normalization_builder()
        self.activation_middle = tf.keras.layers.ReLU()

        # Output 1x1 convolution without activation (is done externally) and 4x as many filters
        self.conv_out = tf.keras.layers.Conv2D(
            filters=filters_out,
            kernel_size=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )
        self.normalization_out = normalization_builder()
        self.activation_out = tf.keras.layers.ReLU()

        # Projection shortcut if required (i.e. when downsampling or changing the number of features)
        self.conv_residual = None
        self.normalization_residual = None
        if downsample or projection_shortcut:
            self.conv_residual = tf.keras.layers.Conv2D(
                filters=filters_out,
                kernel_size=1,
                strides=strides_in,
                padding='same',
                activation=None,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )
            self.normalization_residual = normalization_builder()
            # No activation here, is done on all features on output

    def call(self, inputs, **kwargs):
        # Perform projection shortcut if necessary
        residuals = inputs
        if self.conv_residual is not None:
            assert self.normalization_residual is not None
            residuals = self.conv_residual(residuals)
            residuals = self.normalization_residual(residuals)

        # Differentiate actual block from residual handling
        with tf.keras.backend.name_scope('block'):
            features = self.conv_in(inputs)
            features = self.normalization_in(features)
            features1 = self.activation_in(features)

            features = self.conv_middle(features1)
            features = self.normalization_middle(features)
            features2 = self.activation_middle(features)

            features = self.conv_out(features2)
            block_output = self.normalization_out(features)

        # Add residuals
        pre_activation = block_output + residuals

        # Perform final activation
        output = self.activation_out(pre_activation)

        return output
