import typing

import tensorflow as tf


class BatchNormReLUConvDropout(tf.keras.layers.Layer):
    """
    A helper layer for building FC-Nets. It applies: batch norm -> ReLU -> Conv2D -> Dropout.
    """

    def __init__(
            self,
            kernel_size: int = 3,
            n_features: int = 2,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            name: str = None
    ):
        """
        Args:
            kernel_size (int): Kernel size, defaults to 3.
            n_features (int): Feature maps in output.
            dropout_rate (float): Dropout rate.
            weight_decay (float): Convolutional layer kernel L2 regularisation parameter.
            name (str): Name in tensorflow graph.
        """
        super(BatchNormReLUConvDropout, self).__init__(name=name)

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d = tf.keras.layers.Conv2D(
            n_features,
            kernel_size=kernel_size,
            padding='same',
            strides=1,
            use_bias=True,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.do = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.do(x)
        return x


class TransitionDown(tf.keras.layers.Layer):
    """A helper layer for FC-Nets. It applies: Batch Norm -> ReLU -> Conv2D(1x1) -> Max Pool
    """

    def __init__(
            self,
            filters: int,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            name: str = None
    ):
        """
        Args:
            filters (int): The transition down "layer" keeps the number of feature maps unchanged,
                which is why it cannot have a default value.
            dropout_rate (float): Dropout rate.
            weight_decay (float): Convolutional layer kernel L2 regularisation parameter.
            name (str): Name in tensorflow graph.
        """
        super(TransitionDown, self).__init__(name=name)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d1x1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            padding='same',
            strides=1,
            use_bias=True,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.do = tf.keras.layers.Dropout(dropout_rate)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d1x1(x)
        x = self.do(x)
        x = self.maxpool(x)
        return x


class TransitionUp(tf.keras.layers.Layer):
    """A simple transition up layer for the tiramisu architecture.
    """

    def __init__(
            self,
            filters: int,
            weight_decay: float = 0.2,
            name: str = None
    ):
        """
        Args:
            filters (int): The transition up layer keeps the number of feature maps unchanged, which is why it cannot
                have a default value.
            weight_decay (float): weight decay regularisation parameter.
            name (str): Name in tensorflow graph.
        """
        super(TransitionUp, self).__init__(name=name)
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
    """A dense block for the tiramisu architecture.

    It contains several convolutional layers and concatenations. The input to each convolutional layers is the
    concatenation of all outputs of the previous convolutional layers and the input to the dense block. The
    output of the dense block is a concatenation of all outputs of the convolutional layers in the dense block.
    """

    def __init__(
            self,
            n_layers: int,
            growth_rate: int = 16,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            name: str = None
    ):
        """

        Args:
            n_layers (int): Number of convolutional layers in the dense block.
            growth_rate (int): Growth rate of the dense block, the dense block will output
                n_layers*growth_rate featuremaps.
            dropout_rate (float): Dropout rate.
            weight_decay (float): Weight decay.
            name (str): Name in tensorflow graph.
        """
        super(DenseBlock, self).__init__(name=name)

        self.n_layers = n_layers

        self.dense_block_layers = []
        for i in range(n_layers):
            self.dense_block_layers.append(
                BatchNormReLUConvDropout(
                    n_features=growth_rate,
                    dropout_rate=dropout_rate,
                    weight_decay=weight_decay,
                    kernel_size=3
                )
            )

    def call(self, input_tensor):
        stack = input_tensor
        layer_output = self.dense_block_layers[0](stack)
        outputs = [layer_output]
        for i in range(self.n_layers - 1):
            stack = tf.keras.layers.concatenate([stack, layer_output])
            layer_output = self.dense_block_layers[i + 1](stack)
            outputs.append(layer_output)

        dense_block_output = tf.keras.layers.concatenate(outputs)
        return dense_block_output


class Tiramisu(tf.keras.models.Model):
    """Tiramisu model for image segmentation tasks.

    Tiramisu architecture as proposed by http://arxiv.org/abs/1611.09326 (The One Hundred Layers Tiramisu)
    """

    def __init__(
            self,
            n_classes: int = 1,
            n_initial_features: int = 48,
            n_layers_per_dense_block: typing.List[int] = [4, 5, 7, 10, 12, 15],
            mirror_dense_blocks: bool = True,
            growth_rate: int = 16,
            dropout_rate: float = 0.2,
            weight_decay: float = 1e-4,
            name=None,
    ):
        """
        Args:
            n_classes (int): Number of feature maps in the output. For binary classification.
            n_initial_features (int): The first layer in the network is a normal conv2d layer,
                which is then fed into the rest of the tiramisu. This is the number of feature maps of
                the first layer.
            n_layers_per_dense_block (:obj:`list` of :obj:`int`): This list controls the number of dense
                blocks and how layers each dense block contains. The "middle" dense block in the network
                is the bottleneck and the number of dense blocks in the down and up paths must match.
                Hence, the list must contain an odd number of elements, unless mirror_dense_blocks is
                True (see below). The elements are interpreted in order ([down, ..., bottleneck, up, ...]).
            mirror_dense_blocks (bool): Usually the dense blocks in the down path and up path are mirrored.
                If this variable is True, it suffices to provide the down path and bottleneck in
                n_layers_per_dense_block.
            growth_rate (int): The number of feature maps of the conv2d layers in the dense blocks.
            dropout_rate (float): Dropout rate to be used in all parts of the tiramisu, which defaults to
                0.2 as suggested by the paper.
            weight_decay (float): Convolutional layer kernel L2 regularisation parameter.
            name (str): Name in tensorflow graph.
        """
        super(Tiramisu, self).__init__(name=name)

        # Append reversed (except last element, the bottleneck) to the list
        if mirror_dense_blocks:
            n_layers_per_dense_block += list(reversed(n_layers_per_dense_block[:-1]))

        # Down and up paths must have the same number of dense blocks, plus one bottleneck dense block
        # means we must have an odd number of dense blocks.
        assert (len(n_layers_per_dense_block) % 2) == 1

        self.n_layers = len(n_layers_per_dense_block) // 2

        #########
        # Input #
        #########

        # Initial conv layer that that servers as "input" to the rest of the tiramisu.
        self.in_conv = tf.keras.layers.Conv2D(
            n_initial_features,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer='he_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

        #############
        # Down Path #
        #############
        # each step in the down path contains a dense block and a transition down layer,
        # hence the down_path will contain tuples of dense block - transition down layer pairs.
        self.down_path = []
        n_filters = n_initial_features
        for i in range(self.n_layers):
            n_filters += n_layers_per_dense_block[i] * growth_rate
            dense_block = DenseBlock(
                n_layers=n_layers_per_dense_block[i],
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                name="DB_{}".format(i)
            )
            td = TransitionDown(
                n_filters,
                dropout_rate=dropout_rate,
                name="TD_{}".format(i)
            )
            self.down_path.append((dense_block, td))

        ##############
        # Bottleneck #
        ##############
        # The bottleneck consists of a single dense block.
        # The "middle" number in the list "n_layers_per_dense_block is at index self.n_layers.
        self.dense_block_bottleneck = DenseBlock(
            n_layers=n_layers_per_dense_block[self.n_layers],
            growth_rate=growth_rate,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay,
            name="DB_{}_Bottleneck".format(self.n_layers)
        )

        ###########
        # Up Path #
        ###########
        # The up path consists of pairs of upsampling wth a transposed convolution, followed by concatenation with a
        # skip connection, which is then fed into a dense block. The upsampling layer does not change the number of
        # output features. Thus, the output feature count matches the number of layers in the previous dense block
        # multiplied by the growth rate. (self.n_layers + i is the index of the previous dense block in the list
        # n_layers_per_dense_block)
        self.up_path = []
        for i in range(self.n_layers):
            n_filters = growth_rate * n_layers_per_dense_block[self.n_layers + i]
            transition_up = TransitionUp(
                filters=n_filters,
                weight_decay=weight_decay,
                name="TU_{}".format(i)
            )
            dense_block = DenseBlock(
                n_layers=n_layers_per_dense_block[self.n_layers + i + 1],
                growth_rate=growth_rate,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                name="DB_{}".format(i + self.n_layers + 1)
            )
            self.up_path.append((transition_up, dense_block))

        #######
        # Out #
        #######
        # Lastly, we use one more 1x1 output layer to reduce the number of features to the number of classes of the
        # of the segmentation task.
        self.conv2d1x1 = tf.keras.layers.Conv2D(
            n_classes,
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
        # dense block outputs in the up paths. These "skip connections" are stored in this list of "skips".
        skips = []

        #########
        # Input #
        #########
        # In the down path, the outputs of the all dense block layers plus the input to the dense block are
        # concatenated to one larger input to the following transition down. The stack variable collects the
        # concatenations. Similarly so in the up path, but there the inputs of the dense blocks are not
        # concatenated to the output of the dense blocks.
        stack = self.in_conv(input_tensor)

        #############
        # Down Path #
        #############
        for i in range(self.n_layers):
            dense_block, transition_down = self.down_path[i]
            dense_block_output = dense_block(stack)
            stack = tf.keras.layers.concatenate([stack, dense_block_output])
            skips.append(stack)
            stack = transition_down(stack)

        # reverse the skip connections list for easy handling in up path
        skips = list(reversed(skips))

        ##############
        # Bottleneck #
        ##############
        stack = self.dense_block_bottleneck(stack)

        ###########
        # Up Path #
        ###########
        for i in range(self.n_layers):
            transition_up, dense_block = self.up_path[i]

            upsampled = transition_up(stack)
            skip = skips[i]
            skip_shape = tf.shape(skip)
            resized = tf.image.resize_with_crop_or_pad(upsampled, skip_shape[1], skip_shape[2])
            concated = tf.keras.layers.concatenate([skip, resized])
            stack = dense_block(concated)

        #######
        # Out #
        #######
        stack = self.conv2d1x1(stack)
        return stack


def build_fc_dense_net(
        model: int = 103,
        dropout_rate: float = 0.2,
        weight_decay: float = 1e-4
) -> tf.keras.models.Model:
    """Builds a standard FC-DenseNet[n_layers], where n_layers may be one of 56, 67 103

    Args:
        model (int): May be one of 56, 67, 103, and describes which of the three standard tiramisu models
            is to be built.
        dropout_rate: The dropout rate.
        weight_decay: The weight decay parameter used for normalisation.
    Returns:
        tf.keras.models.Model: A standard FC-DenseNet[n_layers] tiramisu model.
    """
    assert (model in [56, 67, 103])
    n_layers_per_dense_block = {
        56: [4, 4, 4, 4, 4, 4],
        67: [5, 5, 5, 5, 5, 5],
        103: [4, 5, 7, 10, 12, 15]
    }
    growth_rates = {
        56: 12,
        67: 16,
        103: 16
    }

    model = Tiramisu(
        n_initial_features=48,
        n_layers_per_dense_block=n_layers_per_dense_block[model],
        growth_rate=growth_rates[model],
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model
