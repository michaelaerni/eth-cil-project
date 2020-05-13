import typing

import tensorflow as tf

"""
File containing classes to build models as proposed by the paper "The One Hunderd Layers Tiramisu", which can
be found at http://arxiv.org/abs/1611.09326.
"""


class DenseBlockLayer(tf.keras.layers.Layer):
    """
    A helper layer for building FC-Nets. It applies: batch norm -> ReLU -> Conv2D -> Dropout.
    
    Note that this intentionally doesn't follow the traditional layer ordering.
    Usually batch norm and ReLU follow a Conv2D and not vice versa. See the paper for a detailed description.
    """

    def __init__(
            self,
            features: int,
            kernel_size: int = 3,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            **kwargs
    ):
        """
        Args:
            kernel_size: Kernel size, defaults to 3.
            features: Feature maps in output.
            dropout_rate: Dropout rate.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
            kernel_initializer: Initializer used to seed convolution kernel weights.
        """
        super(DenseBlockLayer, self).__init__(**kwargs)

        # FIXME Here we should use statistics over the validation set for batch normalisation, but ther is no obvious
        #  way to imlement this with keras. This is an issue for all batch normalisation layers in the network.
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            features,
            kernel_size=kernel_size,
            padding='same',
            strides=1,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor, **kwargs):
        out = self.batchnorm(input_tensor)
        out = self.relu(out)
        out = self.conv(out)
        out = self.dropout(out)
        return out


class TransitionDown(tf.keras.layers.Layer):
    """
    A helper layer for FC-Nets. It applies: BatchNorm -> ReLU -> Conv (1x1) -> Max Pool
    """

    def __init__(
            self,
            filters: int,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            **kwargs
    ):
        """
        Args:
            filters: The number of filters in the input/output. TransitionDown layers leave the number of feature
                maps unchanged.
            dropout_rate: Dropout rate.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
            kernel_initializer: Initializer used to seed convolution kernel weights.
        """
        super(TransitionDown, self).__init__(**kwargs)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.output_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            strides=1,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, input_tensor, **kwargs):
        out = self.batchnorm(input_tensor)
        out = self.relu(out)
        out = self.output_conv(out)
        out = self.dropout(out)
        out = self.maxpool(out)
        return out


class TransitionUp(tf.keras.layers.Layer):
    """
    A simple TransitionUp layer for the Tiramisu architecture.
    """

    def __init__(
            self,
            filters: int,
            weight_decay: float = 0.2,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            **kwargs
    ):
        """
        Args:
            filters: The number of filters in the input/output. TransitionDown layers leave the number of feature
                maps unchanged.
            weight_decay: Weight decay regularisation parameter.
            kernel_initializer: Initializer used to seed convolution kernel weights.
        """
        super(TransitionUp, self).__init__(**kwargs)
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            padding='same',
            kernel_size=3,
            strides=2,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, input_tensor, **kwargs):
        return self.transposed_conv(input_tensor)


class DenseBlock(tf.keras.layers.Layer):
    """
    A DenseBlock consists of a series of DenseBlockLayers.

    The input to the first DenseBlockLayer is just the input to the DenseBlock. The input of the i-th DenseBlockLayer
    is the output of the (i-1)-th DenseBlockLayer concatenated with the input to the (i-1)-th DenseBlockLayer.

    The output of the entire DenseBlock is the concatenation the
    outputs of all DenseBlockLayers.
    """

    def __init__(
            self,
            layers: int,
            growth_rate: int,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            **kwargs
    ):
        """
        Args:
            layers: Number of convolutional layers in the DenseBlock.
            growth_rate: Growth rate of the DenseBlock, the DenseBlock will output layers*growth_rate feature maps.
            dropout_rate: Dropout rate.
            weight_decay: Weight decay.
            kernel_initializer: Initializer used to seed convolution kernel weights.
        """
        super(DenseBlock, self).__init__(**kwargs)

        # The DenseBlock contains multiple connected DenseBlockLayer layers.
        self.dense_block_layers = []
        for idx in range(layers):
            self.dense_block_layers.append(
                DenseBlockLayer(
                    features=growth_rate,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay,
                    kernel_size=3,
                    kernel_initializer=kernel_initializer
                )
            )

    def call(self, input_tensor, **kwargs):
        features = input_tensor

        layer_output = self.dense_block_layers[0](features)
        outputs = [layer_output]
        for layer in self.dense_block_layers[1:]:
            # Concatenate the input to the output. Note that the features variable is only used as input, never directly
            # added to the outputs list. This is what ensures at most linear growth in the number of feature maps.
            features = tf.concat([features, layer_output], -1)
            layer_output = layer(features)
            outputs.append(layer_output)

        # The final output is now a concatenation of all DenseBlock outputs.
        dense_block_output = tf.concat(outputs, -1)
        return dense_block_output


class Tiramisu(tf.keras.models.Model):
    """
    Tiramisu architecture as proposed by http://arxiv.org/abs/1611.09326 (The One Hundred Layers Tiramisu)
    """

    _DEFAULT_INITIAL_FILTERS = 48
    """
    Every FC-DenseNet described by the paper uses 48 filters in the first convoluitonal layer.
    """

    _KERNEL_INITIALIZER = 'he_uniform'
    """
    The paper uses the he_uniform initializer for all kernels of convolutional layers.
    """

    def __init__(
            self,
            growth_rate: int,
            layers_per_dense_block: typing.List[int],
            layers_bottleneck: int,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            initial_filters: typing.Optional[int] = None,
            **kwargs
    ):
        """
        Args:
            growth_rate: The number of feature maps of the DenseBlockLayers in the DenseBlock.
            layers_per_dense_block: The number of DenseBlockLayers per DenseBlock in the down path and up path.
                This is mirrored in the up path.
            layers_bottleneck: The number of DenseBlockLayers in the bottleneck DenseBlock.
            dropout_rate: Dropout rate to be used in all parts of the Tiramisu, which defaults to 0.2
                as suggested by the paper.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
            initial_filters: Number of filters of the first convolutional layer.
        """
        super(Tiramisu, self).__init__(**kwargs)

        layers_per_dense_block_down = layers_per_dense_block
        layers_per_dense_block_up = list(reversed(layers_per_dense_block))

        if initial_filters is None:
            initial_filters = self._DEFAULT_INITIAL_FILTERS

        filters = initial_filters
        self.in_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self._KERNEL_INITIALIZER,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

        # Each step in the down path consists of a DenseBlock and a TransitionDown stored as tuples in down_path.
        self.down_path = []
        for layers_dense_block in layers_per_dense_block_down:
            filters += layers_dense_block * growth_rate
            dense_block = DenseBlock(
                layers=layers_dense_block,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                kernel_initializer=self._KERNEL_INITIALIZER
            )
            transition_down = TransitionDown(
                filters,
                dropout_rate=dropout_rate
            )
            self.down_path.append((dense_block, transition_down))

        self.dense_block_bottleneck = DenseBlock(
            layers=layers_bottleneck,
            growth_rate=growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            kernel_initializer=self._KERNEL_INITIALIZER
        )

        # Each step in the up path consists of a TransitionUp and a DenseBlock, stored as tuples in up_path.
        self.up_path = []
        for layers_dense_block in layers_per_dense_block_up:
            filters = growth_rate * layers_dense_block
            transition_up = TransitionUp(
                filters=filters,
                weight_decay=weight_decay,
                kernel_initializer=self._KERNEL_INITIALIZER
            )
            dense_block = DenseBlock(
                layers=layers_dense_block,
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                kernel_initializer=self._KERNEL_INITIALIZER
            )
            self.up_path.append((transition_up, dense_block))

        # Reduces the number of feature maps to the number of classes.
        self.conv_featuremaps_to_classes = tf.keras.layers.Conv2D(
            1,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            activation=None,
            kernel_initializer=self._KERNEL_INITIALIZER,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, input_tensor, **kwargs):


        # down_path_features collects the increasing features in the down path.
        down_path_features = self.in_conv(input_tensor, name='input_conv')

        # Down path
        # The outputs of the DenseBlocks, concatenated with the inputs, are concatenated to the matching upsampled
        # DenseBlock outputs in the up path, forming a "skip connection". These tensors are stored in the "skips" list.
        skips = []
        with tf.keras.backend.name_scope('down_path'):
            for dense_block, transition_down in self.down_path:
                down_dense_block_out = dense_block(down_path_features)
                down_path_features = tf.concat([down_path_features, down_dense_block_out], -1)
                skips.append(down_path_features)
                down_path_features = transition_down(down_path_features)

        skips = list(reversed(skips))

        # Bottleneck
        up_dense_block_out = self.dense_block_bottleneck(down_path_features)

        # Up path
        with tf.keras.backend.name_scope('up_path'):
            for (transition_up, dense_block), skip in zip(self.up_path, skips):
                upsampled = transition_up(up_dense_block_out)
                skip_shape = tf.shape(skip)

                # FIXME: We crop the upsampled tensors because we would otherwise end up with an output segmentation mask of
                #  different spacial dimensions to the input image, but this is somewhat wasteful.
                cropped = tf.image.resize_with_crop_or_pad(upsampled, skip_shape[1], skip_shape[2])
                concated = tf.concat([skip, cropped], -1)
                up_dense_block_out = dense_block(concated)

        # Reduce to classes for output
        out = self.conv_featuremaps_to_classes(up_dense_block_out, name='output_conv')
        return out


class TiramisuFCDenseNet56(Tiramisu):
    """
    A standard FC-DenseNet56, based on the Tiramisu architecture.
    """

    def __init__(
            self,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4
    ):
        """
        Args:
            dropout_rate: The dropout rate.
            weight_decay: The weight decay.
        """
        super(TiramisuFCDenseNet56, self).__init__(
            growth_rate=12,
            layers_per_dense_block=[4, 4, 4, 4, 4],
            layers_bottleneck=4,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )


class TiramisuFCDenseNet67(Tiramisu):
    """
    A standard FC-DenseNet67, based on the Tiramisu architecture.
    """

    def __init__(
            self,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4
    ):
        """
        Args:
            dropout_rate: The dropout rate.
            weight_decay: The weight decay.
        """
        super(TiramisuFCDenseNet67, self).__init__(
            growth_rate=16,
            layers_per_dense_block=[5, 5, 5, 5, 5],
            layers_bottleneck=5,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )


class TiramisuFCDenseNet103(Tiramisu):
    """
    A standard FC-DenseNet103, based on the Tiramisu architecture.
    """

    def __init__(
            self,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4
    ):
        """
        Args:
            dropout_rate: The dropout rate.
            weight_decay: The weight decay.
        """
        super(TiramisuFCDenseNet103, self).__init__(
            growth_rate=16,
            layers_per_dense_block=[4, 5, 7, 10, 12],
            layers_bottleneck=15,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )
