import typing

import tensorflow as tf


class Layer2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 3, n_out_features: int = 2, dropout_rate: float = 0.2):
        super(Layer2D, self).__init__()
        self.kernel_size = kernel_size
        self.n_out_features = n_out_features
        self.dropout_rate = dropout_rate

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d = tf.keras.layers.Conv2D(self.n_out_features,
                                          kernel_size=self.kernel_size,
                                          padding='same',
                                          strides=1,
                                          use_bias=True)
        self.do = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d(x)
        x = self.do(x)
        return x


class TransitionDown(tf.keras.layers.Layer):
    def __init__(self, filters, dropout_rate=0.2):
        super(TransitionDown, self).__init__()
        self.dropout_rate = dropout_rate
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2d1x1 = tf.keras.layers.Conv2D(filters,
                                             kernel_size=1,
                                             padding='same',
                                             strides=1,
                                             use_bias=True)
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=2, padding='same')

    def call(self, input_tensor):
        x = self.bn(input_tensor)
        x = self.relu(x)
        x = self.conv2d1x1(x)
        x = self.maxpool(x)
        return x


class Tiramisu(tf.keras.models.Model):
    def __init__(self,
                 n_initial_features: int = 48,
                 n_denseblock_layers: typing.List[int] = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4],
                 growth_rate: int = 16,
                 dropout_rate: float = 0.2):
        super(Tiramisu, self).__init__()
        self.n_layers = len(n_denseblock_layers) // 2

        self.in_conv = tf.keras.layers.Conv2D(
            n_initial_features,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=True
        )

        #############
        # Down Path #
        #############
        self.down_path = []
        n_filters = n_initial_features
        for i in range(self.n_layers):
            n_filters += n_denseblock_layers[i] * growth_rate
            db_layers = []
            # Dense Block
            for j in range(n_denseblock_layers[i]):
                db_layers.append(Layer2D(kernel_size=3, n_out_features=growth_rate, dropout_rate=dropout_rate))
            td = TransitionDown(n_filters, dropout_rate=dropout_rate)
            self.down_path.append((db_layers, td))

        ##############
        # Bottleneck #
        ##############
        self.db_bottleneck = []
        # Dense Block
        for i in range(n_denseblock_layers[self.n_layers]):
            self.db_bottleneck.append(Layer2D(kernel_size=3, n_out_features=growth_rate, dropout_rate=dropout_rate))

        ###########
        # Up Path #
        ###########
        self.up_path = []
        for i in range(self.n_layers):
            n_filters = growth_rate * n_denseblock_layers[self.n_layers + i + 1]
            transp_layer = tf.keras.layers.Conv2DTranspose(filters=n_filters,
                                                           padding='same',
                                                           kernel_size=3,
                                                           strides=2)
            # Dense Block
            db_layers = []
            for j in range(n_denseblock_layers[self.n_layers + i]):
                db_layers.append(Layer2D(kernel_size=3, n_out_features=growth_rate, dropout_rate=dropout_rate))

            self.up_path.append((transp_layer, db_layers))

        #######
        # Out #
        #######
        self.conv2d16x16 = tf.keras.layers.Conv2D(1,
                                               kernel_size=16,
                                               strides=16,
                                               padding='same',
                                               use_bias=True,
                                               activation=None)

    def call(self, input_tensor):
        skips = []

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

        #######
        # Out #
        #######
        stack = self.conv2d16x16(stack)
        return stack
