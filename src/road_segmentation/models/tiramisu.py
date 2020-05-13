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
            **kwargs
    ):
        """
        Args:
            kernel_size: Kernel size, defaults to 3.
            features: Feature maps in output.
            dropout_rate: Dropout rate.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
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
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor):
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
            **kwargs
    ):
        """
        Args:
            filters: The number of filters in the input/output. TransitionDown layers leave the number of feature
                maps unchanged.
            dropout_rate: Dropout rate.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
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
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, input_tensor):
        out = self.batchnorm(input_tensor)
        out = self.relu(out)
        out = self.output_conv(out)
        out = self.dropout(out)
        out = self.maxpool(out)
        return out


class TransitionUp(tf.keras.layers.Layer):
    """
    A simple transition up layer for the tiramisu architecture.
    """

    def __init__(
            self,
            filters: int,
            weight_decay: float = 0.2,
            **kwargs
    ):
        """
        Args:
            filters: The number of filters in the input/output. TransitionDown layers leave the number of feature
                maps unchanged.
            weight_decay: Weight decay regularisation parameter.
        """
        super(TransitionUp, self).__init__(**kwargs)
        self.transposed_conv = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            padding='same',
            kernel_size=3,
            strides=2,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, input_tensor):
        return self.transposed_conv(input_tensor)


class DenseBlock(tf.keras.layers.Layer):
    """
    A DenseBlock consists of a series of DenseBlockLayers.

    The input to the first DenseBlockLayer is just the input to the DenseBlock. The input of the
    i-th DenseBlockLayer is the output of the (i-1)-th DenseBlockLayer concatenated with the input
    to the (i-1)-th DenseBlockLayer.

    The output of the entire DenseBlock is the concatenation the
    outputs of all DenseBlockLayers.
    """

    def __init__(
            self,
            layers: int,
            growth_rate: int,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            **kwargs
    ):
        """
        Args:
            layers: Number of convolutional layers in the dense block.
            growth_rate: Growth rate of the dense block, the dense block will output
                layers*growth_rate feature maps.
            dropout_rate: Dropout rate.
            weight_decay: Weight decay.
        """
        super(DenseBlock, self).__init__(**kwargs)

        # The dense block contains multiple connected DenseBlockLayer layers.
        self.dense_block_layers = []
        for idx in range(layers):
            self.dense_block_layers.append(
                DenseBlockLayer(
                    features=growth_rate,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay,
                    kernel_size=3
                )
            )

    def call(self, input_tensor):
        features = input_tensor

        layer_output = self.dense_block_layers[0](features)
        outputs = [layer_output]
        for layer in self.dense_block_layers[1:]:
            # Concatenate the input to the output. Note that the features variable is only used as input, never directly
            # added to the outputs list. This is what ensures at most linear growth in the number of feature maps.
            features = tf.concat([features, layer_output], -1)
            layer_output = layer(features)
            outputs.append(layer_output)

        # The final output is now a concatenation of all dense block layer outputs.
        dense_block_output = tf.concat(outputs, -1)
        return dense_block_output


class Tiramisu(tf.keras.models.Model):
    """
    Tiramisu architecture as proposed by http://arxiv.org/abs/1611.09326 (The One Hundred Layers Tiramisu)
    """

    def __init__(
            self,
            growth_rate: int,
            layers_per_dense_block: typing.List[int],
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            **kwargs
    ):
        """
        Args:
            growth_rate: The number of feature maps of the convolutional layers in the dense blocks.
            layers_per_dense_block: This list controls the number of dense blocks and how layers each
                dense block contains. The "middle" dense block in the network is the bottleneck and
                the number of dense blocks in the down and up paths must match. Hence, the list must
                contain an odd number of elements, unless mirror_dense_blocks is True (see below).
                The elements are interpreted in order ([down, ..., bottleneck]). The dense blocks
                in the down path are automatically mirrored.
            dropout_rate: Dropout rate to be used in all parts of the tiramisu, which defaults to 0.2
                as suggested by the paper.
            weight_decay: Convolutional layer kernel L2 regularisation parameter.
        """
        super(Tiramisu, self).__init__(**kwargs)

        # Append reversed (except last element, the bottleneck) to the list
        layers_per_dense_block += list(reversed(layers_per_dense_block[:-1]))

        # Input:
        # Initial conv layer that that servers as "input" to the rest of the tiramisu.
        filters = 48
        self.in_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

        # Down Path:
        # Each step in the down path contains a dense block and a transition down layer,
        # hence the down_path will contain tuples of dense block - transition down layer pairs.
        self.down_path = []

        # The down and up paths consist of half the blocks in the list minus the bottleneck block.
        path_length = len(layers_per_dense_block) // 2
        for dense_block_idx in range(path_length):
            filters += layers_per_dense_block[dense_block_idx] * growth_rate
            dense_block = DenseBlock(
                layers=layers_per_dense_block[dense_block_idx],
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay
            )
            transition_down = TransitionDown(
                filters,
                dropout_rate=dropout_rate
            )
            self.down_path.append((dense_block, transition_down))

        # Bottleneck:
        # The bottleneck consists of a single dense block.
        # The "middle" number in the list "layers_per_dense_block is at index path_length.
        self.dense_block_bottleneck = DenseBlock(
            layers=layers_per_dense_block[path_length],
            growth_rate=growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )

        # Up Path:
        # The up path consists of pairs of upsampling wth a transposed convolution, followed by concatenation with a
        # skip connection, which is then fed into a dense block.
        self.up_path = []
        for dense_block_idx_zero in range(path_length):
            dense_block_idx = dense_block_idx_zero + path_length

            filters = growth_rate * layers_per_dense_block[dense_block_idx]
            transition_up = TransitionUp(
                filters=filters,
                weight_decay=weight_decay
            )
            dense_block = DenseBlock(
                layers=layers_per_dense_block[dense_block_idx + 1],
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay
            )
            self.up_path.append((transition_up, dense_block))

        # Lastly, we use one more 1x1 convolutional layer to reduce the number of features to the number of classes of
        # the segmentation task.
        self.conv_featuremaps_to_classes = tf.keras.layers.Conv2D(
            1,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            activation=None,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, input_tensor):

        # The outputs of the dense blocks, concatenated with the inputs, are concatenated to the matching upsampled
        # dense block outputs in the up path, forming a "skip connection". These tensors are stored in the "skips" list.
        skips = []

        # In the down path, the outputs of the all dense block layers plus the input to the dense block are
        # concatenated to one larger input to the following transition down. The down_path_features variable collects
        # the concatenations. Similarly so in the up path, but there the inputs of the dense blocks are not
        # concatenated to the output of the dense blocks.
        down_path_features = self.in_conv(input_tensor)

        # Down Path
        for dense_block, transition_down in self.down_path:
            down_dense_block_out = dense_block(down_path_features)
            down_path_features = tf.concat([down_path_features, down_dense_block_out], -1)
            skips.append(down_path_features)
            down_path_features = transition_down(down_path_features)

        skips = list(reversed(skips))

        up_dense_block_out = self.dense_block_bottleneck(down_path_features)

        # Up Path
        for (transition_up, dense_block), skip in zip(self.up_path, skips):
            upsampled = transition_up(up_dense_block_out)
            skip_shape = tf.shape(skip)

            # FIXME: We crop the upsampled tensors because we would otherwise end up with an output segmentation mask of
            #  different spacial dimensions to the input image, but this is somewhat wasteful.
            cropped = tf.image.resize_with_crop_or_pad(upsampled, skip_shape[1], skip_shape[2])
            concated = tf.concat([skip, cropped], -1)
            up_dense_block_out = dense_block(concated)

        out = self.conv_featuremaps_to_classes(up_dense_block_out)
        return out


class TiramisuFCDenseNet56(Tiramisu):
    """
    A standard FC-DenseNet56, based on the tiramisu architecture.
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
            layers_per_dense_block=[4, 4, 4, 4, 4, 4],
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )


class TiramisuFCDenseNet67(Tiramisu):
    """
    A standard FC-DenseNet67, based on the tiramisu architecture.
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
            layers_per_dense_block=[5, 5, 5, 5, 5, 5],
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )


class TiramisuFCDenseNet103(Tiramisu):
    """
    A standard FC-DenseNet103, based on the tiramisu architecture.
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
            layers_per_dense_block=[4, 5, 7, 10, 12, 15],
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )
