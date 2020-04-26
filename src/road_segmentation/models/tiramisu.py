import typing

import tensorflow as tf


class Layer2D(tf.keras.layers.Layer):
    """
    A helper layer for building FC-Nets. It applies: batch norm -> ReLU -> Conv2D -> Dropout.
    """

    def __init__(self, kernel_size: int = 3, n_features: int = 2, dropout_rate: float = 0.2,
                 weight_decay: float = 1e-4):
        """

        Args:
            :param kernel_size: Kernel size, defaults to 3.
            :param n_features: Feature maps in output.
            :param dropout_rate: Dropout rate.
            :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
        """
        super(Layer2D, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d = tf.keras.layers.Conv2D(n_features,
                                             kernel_size=kernel_size,
                                             padding='same',
                                             strides=1,
                                             use_bias=True,
                                             kernel_initializer='he_uniform',
                                             kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.do = tf.keras.layers.Dropout(dropout_rate)

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.do(x)
        return x


class TransitionDown(tf.keras.layers.Layer):
    """
    A helper layer for FC-Nets. It applies: Batch Norm -> ReLU -> Conv2D(1x1) -> Max Pool
    """

    def __init__(self, filters: int, dropout_rate: float = 0.2, weight_decay: float = 1e-4):
        """
        Args:
            :param filters: The transition down "layer" keeps the number of feature maps unchanged,
             which is why it cannot have a default value.
            :param dropout_rate: Dropout rate.
            :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
        """
        super(TransitionDown, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d1x1 = tf.keras.layers.Conv2D(filters,
                                                kernel_size=1,
                                                padding='same',
                                                strides=1,
                                                use_bias=True,
                                                kernel_initializer='he_uniform',
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.do = tf.keras.layers.Dropout(dropout_rate)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d1x1(x)
        x = self.do(x)
        x = self.maxpool(x)
        return x


class Tiramisu(tf.keras.models.Model):
    """
    Tiramisu architecture as proposed by http://arxiv.org/abs/1611.09326 (The One Hundred Layers Tiramisu)
    """

    def __init__(self,
                 n_classes: int = 1,
                 n_initial_features: int = 48,
                 n_layers_per_dense_block: typing.List[int] = [4, 5, 7, 10, 12, 15],
                 mirror_dense_blocks: bool = True,
                 growth_rate: int = 16,
                 dropout_rate: float = 0.2,
                 weight_decay: float = 1e-4):
        """
        Args:
            :param n_classes: Number of feature maps in the output. For binary classification.

            :param n_initial_features: The first layer in the network is a normal conv2d layer,
             which is then fed into the rest of the tiramisu. This is the number of feature
             maps of the first layer.

            :param n_layers_per_dense_block: This list controls the number of dense blocks and
             how layers each dense block contains. The "middle" dense block in the network is
             the bottleneck and the number of dense blocks in the down and up paths must match.
             Hence, the list must contain an odd number of elements, unless mirror_dense_blocks
             is True (see below).
             The elements are interpreted in order ([down, ..., bottleneck, up, ...]).

            :param mirror_dense_blocks: Usually the dense blocks in the down path and up path
             are mirrored. If this variable is True, it suffices to provide the down path and
             bottleneck in n_layers_per_dense_block.

            :param growth_rate: The number of feature maps of the conv2d layers in the dense
             blocks.

            :param dropout_rate: Dropout rate to be used in all parts of the tiramisu, which
             defaults to 0.2 as suggested by the paper.

            :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
        """
        super(Tiramisu, self).__init__()

        # Append reversed (except last element, the bottleneck) to the list
        if mirror_dense_blocks:
            n_layers_per_dense_block += n_layers_per_dense_block[:-1][::-1]

        assert (len(n_layers_per_dense_block) % 2) == 1

        self.n_layers = len(n_layers_per_dense_block) // 2

        #########
        # Input #
        #########
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
        self.down_path = []
        n_filters = n_initial_features
        for i in range(self.n_layers):
            n_filters += n_layers_per_dense_block[i] * growth_rate
            db_layers = []
            # Dense Block
            for j in range(n_layers_per_dense_block[i]):
                db_layers.append(Layer2D(kernel_size=3, n_features=growth_rate, dropout_rate=dropout_rate,
                                         weight_decay=weight_decay))
            td = TransitionDown(n_filters, dropout_rate=dropout_rate)
            self.down_path.append((db_layers, td))

        ##############
        # Bottleneck #
        ##############
        self.db_bottleneck = []
        # Dense Block
        for i in range(n_layers_per_dense_block[self.n_layers]):
            self.db_bottleneck.append(
                Layer2D(kernel_size=3, n_features=growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay))

        ###########
        # Up Path #
        ###########
        self.up_path = []
        for i in range(self.n_layers):
            n_filters = growth_rate * n_layers_per_dense_block[self.n_layers + i]
            transp_layer = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                                           padding='same',
                                                           kernel_size=3,
                                                           strides=2,
                                                           kernel_initializer='he_uniform',
                                                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
            # Dense Block
            db_layers = []
            for j in range(n_layers_per_dense_block[self.n_layers + i + 1]):
                db_layers.append(Layer2D(kernel_size=3,
                                         n_features=growth_rate,
                                         dropout_rate=dropout_rate,
                                         weight_decay=weight_decay))

            self.up_path.append((transp_layer, db_layers))

        #######
        # Out #
        #######
        self.conv2d16x16 = tf.keras.layers.Conv2D(n_classes,
                                                  kernel_size=1,
                                                  strides=1,
                                                  padding='same',
                                                  use_bias=True,
                                                  activation=None,
                                                  kernel_initializer='he_uniform',
                                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay))

    def call(self, input_tensor):
        skips = []

        #########
        # Input #
        #########
        stack = self.in_conv(input_tensor)

        #############
        # Down Path #
        #############
        for i in range(self.n_layers):
            db, td = self.down_path[i]
            with tf.name_scope("DB_{}".format(i)):
                for l in db:
                    o = l(stack)
                    stack = tf.keras.layers.concatenate([stack, o])
                skips.append(stack)

            with tf.name_scope("TD_{}".format(i)):
                stack = td(stack)

        skips = skips[::-1]

        ##############
        # Bottleneck #
        ##############
        to_upsample = []
        with tf.name_scope("bottleneck"):
            for l in self.db_bottleneck:
                o = l(stack)
                to_upsample.append(o)
                stack = tf.keras.layers.concatenate([stack, o])

        ###########
        # Up Path #
        ###########
        for i in range(self.n_layers):
            i_ = i + self.n_layers + 1
            transp, db = self.up_path[i]
            with tf.name_scope("TU_{}".format(i_)):
                o = tf.keras.layers.concatenate(to_upsample)
                o = transp(o)
                skip = skips[i]
                skip_shape = tf.shape(skip)
                o = tf.image.resize_with_crop_or_pad(o, skip_shape[1], skip_shape[2])
                stack = tf.keras.layers.concatenate([o, skip])

            to_upsample = []
            with tf.name_scope("DB_{}".format(i_)):
                for l in db:
                    o = l(stack)
                    to_upsample.append(o)
                    stack = tf.keras.layers.concatenate([stack, o])
                    # The avid code reviewer might notice at this point, that the last concatenation
                    # is discarded (except after the very last dense block) and thus seemingly
                    # completely unnecessary. It is, however, essential that we always compute this,
                    # because otherwise the stated number of feature maps in the paper would be
                    # incorrect, despite the fact that they already "corrected" those calculations
                    # at least once. So, in order for those calculations to be correct, at certain
                    # points in the network there needs to exist some number of feature maps and by
                    # always concatenating we ensure that number of feature maps to exist at the
                    # correct points in the network. They need not be in the same tensor, they just
                    # need to exist. Most of them are just discarded afterwards anyway. This surely
                    # is as intended, and not an oversight or just laziness. </sarcasm>
                    # Either way, I feel justified to be lazy too.

        #######
        # Out #
        #######
        stack = self.conv2d16x16(stack)
        return stack


def build_FCDenseNet103(dropout_rate: float = 0.2, weight_decay: float = 1e-4) -> tf.keras.models.Model:
    """
    Builds the standard FC-DenseNet103 as described in the One Hunderd Layers Tiramisu paper.

    Args:
        :param dropout_rate: The dropout rate to be used in the entire tiramisu.
        :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
    Returns:
        :return: The model, ready to be compiled and fitted.
    """
    model = Tiramisu(
        n_initial_features=48,
        n_layers_per_dense_block=[4, 5, 7, 10, 12, 15],
        growth_rate=16,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model


def build_FCDenseNet67(dropout_rate: float = 0.2, weight_decay: float = 1e-4) -> tf.keras.Model:
    """
    Builds the standard FC-DenseNet67 as "described" in the One Hunderd Layers Tiramisu paper.

    Args:
        :param dropout_rate: The dropout rate to be used in the entire tiramisu.
        :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
    Returns:
        :return: The model, ready to be compiled and fitted.
    """
    model = Tiramisu(
        n_initial_features=48,
        n_layers_per_dense_block=[5, 5, 5, 5, 5, 5],
        growth_rate=16,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model


def build_FCDenseNet56(dropout_rate: float = 0.2, weight_decay: float = 1e-4) -> tf.keras.Model:
    """
    Builds the standard FC-DenseNet56 as "described" in the One Hunderd Layers Tiramisu paper.

    Args:
        :param dropout_rate: The dropout rate to be used in the entire tiramisu.
        :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
    Returns:
        :return: The model, ready to be compiled and fitted.
    """
    model = Tiramisu(
        n_initial_features=48,
        n_layers_per_dense_block=[4, 4, 4, 4, 4, 4],
        growth_rate=12,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model


def build_FCDenseNetTiny(dropout_rate: float = 0.2, weight_decay: float = 1e-4) -> tf.keras.models.Model:
    """
    Builds a tiny tiramisu which can be useful for testing, but most likely useless
    for actual segmentation.

    Args:
        :param dropout_rate: The dropout rate to be used in the entire tiramisu.
        :param weight_decay: Convolutional layer kernel L2 regularisation parameter.
    Returns:
        :return: The model, ready to be compiled and fitted.
    """
    model = Tiramisu(
        n_initial_features=4,
        n_layers_per_dense_block=[2, 2, 2],
        growth_rate=2,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay
    )
    return model
