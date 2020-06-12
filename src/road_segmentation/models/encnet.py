import typing

import tensorflow as tf

"""
Implementation of layers to build EncNets according to (http://arxiv.org/abs/1803.08904)
"""


class EncNet(tf.keras.models.Model):
    """
    TODO: Documentation
    """

    def __init__(
            self,
            classes=1,
            down_path_length=2,
            codewords=32,
            dropout_rate=0.2,
            weight_decay=1e-4
    ):
        super(EncNet, self).__init__()

        # TODO: Replace dummy network with proper ResNet/FastFCN as in paper.
        features = 32
        self.in_conv = tf.keras.layers.Conv2D(
            features,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

        self.down_path = []
        for _ in range(down_path_length):
            features = features * 2
            conv = tf.keras.layers.Conv2D(
                features,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            )
            dropout = tf.keras.layers.Dropout(dropout_rate)
            relu = tf.keras.layers.ReLU()
            pool = tf.keras.layers.MaxPool2D(
                pool_size=2,
                padding='same'
            )
            self.down_path.append((conv, dropout, relu, pool))

        self.context_encoding_module = ContextEncodingModule(
            codewords=codewords,
            classes=classes
        )

        self.up_path = []

        for _ in range(down_path_length):
            conv = tf.keras.layers.Conv2DTranspose(
                filters=features,
                kernel_size=3,
                padding='same',
                strides=2,
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
            )
            dropout = tf.keras.layers.Dropout(dropout_rate)
            relu = tf.keras.layers.ReLU()
            features = features // 2
            self.up_path.append((conv, dropout, relu))

        self.out_conv = tf.keras.layers.Conv2D(
            classes,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            name="segmentation"
        )

    def call(self, inputs, training=False, **kwargs):
        in_shape = tf.shape(inputs)
        features = self.in_conv(inputs)
        skips = []
        for conv, dropout, relu, pool in self.down_path:
            features = conv(features)
            features = dropout(features)
            features = relu(features)
            skips.append(features)
            features = pool(features)

        features, se_loss_output = self.context_encoding_module(features)

        skips = list(reversed(skips))
        for (conv, dropout, relu), skip in zip(self.up_path, skips):
            features = relu(dropout(conv(features)))
            skip_shape = tf.shape(skip)
            features = tf.image.resize_with_crop_or_pad(features, skip_shape[1], skip_shape[2])
            features = tf.concat([skip, features], -1)

        # features = self.upsample(features)
        segmentation_output = tf.image.resize_with_crop_or_pad(features, in_shape[1], in_shape[2])
        segmentation_output = self.out_conv(segmentation_output)

        return segmentation_output, se_loss_output


class ContextEncodingModule(tf.keras.layers.Layer):
    """
    TODO: Documentation
    FIXME: Initialisation of fully connected layers should be done as suggested by paper.
    """

    def __init__(
            self,
            codewords: int,
            classes: int,
            **kwargs
    ):
        """
        Args:
            codewords: Number of codewords to be used.
            features: Dimension (length) of codewords.
            classes: The number of classes in the segmentation task.
        """
        super(ContextEncodingModule, self).__init__(**kwargs)

        self.encoder = Encoder(codewords)

        self.fully_connected_encoding = None

        # No activation for se loss output
        self.fully_connected_se_loss = tf.keras.layers.Dense(
            classes,
            activation='sigmoid'
        )

    def build(self, input_shape):
        features = input_shape[-1]
        self.fully_connected_encoding = tf.keras.layers.Dense(
            features,
            activation='sigmoid'
        )

    def call(self, inputs, **kwargs):
        encodings = self.encoder(inputs)
        featuremaps_attention = self.fully_connected_encoding(encodings)

        # TODO: Check if shapes match: This operation works if "featuremaps_attention" is of 3 dimensions and inputs
        #       has 4 dimensions, but is only correct if "featuremaps_attention" is 4 dimensions as well!
        featuremaps_attention_shape = tf.shape(featuremaps_attention)
        newshape = (featuremaps_attention_shape[0], 1, 1, featuremaps_attention_shape[1])
        featuremaps_attention = tf.reshape(featuremaps_attention, newshape)
        featuremaps_output = featuremaps_attention * inputs

        se_loss_output = self.fully_connected_se_loss(encodings)

        return featuremaps_output, se_loss_output


class Encoder(tf.keras.layers.Layer):
    """
    TODO: Documentation
    FIXME: Refactor possibly into multiple sub classes, what makes sense is TBD.
    """

    def __init__(
            self,
            codewords: int
    ):
        """
        Args:
            codewords: Number of codewords to be used.
        """

        super(Encoder, self).__init__()

        # FIXME: We need the number of codewords in the call function, which results in a name conflict of the
        #  codewords tensor with the actual number of codewords. Hence, breaking the naming convention here.
        self.n_codewords = codewords
        self.features = None

        self.codewords = None

        smoothing_factors_initializer = tf.random_uniform_initializer(
            minval=-1,
            maxval=0
        )
        self.smoothing_factors = self.add_weight(
            name='smoothing_factors',
            shape=(1, 1, self.n_codewords),
            dtype=tf.float32,
            initializer=smoothing_factors_initializer,
            trainable=True
        )

        # Want to apply batch norm on the channels axis, not on the codewords axis
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-2)
        self.relu = tf.keras.layers.ReLU()

    def build(self, input_shape):
        self.features = input_shape[-1]

        # Initialize codewords variables, according to implementation of main author at
        # https://github.com/zhanghang1989/PyTorch-Encoding/blob/d9dea1724e38362a7c75ca9498f595248f283f00/encoding/nn/encoding.py#L86
        support = 1. / ((self.n_codewords * input_shape[-1]) ** (1 / 2))

        codewords_initializer = tf.random_uniform_initializer(
            minval=-support,
            maxval=support
        )

        self.codewords = self.add_weight(
            name='codewords',
            shape=(1, 1, self.n_codewords, self.features),
            dtype=tf.float32,
            initializer=codewords_initializer,
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # TODO: This code needs to be tested thoroughly!
        # Assuming that the input shape is (Batch * width * height * features)
        in_shape = tf.shape(inputs)

        # Number of "pixels" = width * height
        n = in_shape[1] * in_shape[2]

        features_shape = (in_shape[0], n, 1, self.features)

        # (B x n x 1 x C) <= (B x W x H x C)
        features = tf.reshape(inputs, shape=features_shape)

        # (B x n x K x C) <= (B x n x 1 x C) - (1 x 1 x K x C)
        # Tensorflow correctly does pairwise subtraction.
        residuals = features - self.codewords

        # Squared norm
        # (B x n x K) <= reduce_sum(square( (B x n x K x C) ))
        residual_sqnorms = tf.reduce_sum(tf.square(residuals), axis=-1)

        # In the paper, this also is negated. This is done by initialising the smoothing factors
        # with negative values.
        # (B x n x K) <= (B x n x K) * (1 x 1 x K)
        smoothed_sqnorms = residual_sqnorms * self.smoothing_factors

        # (B x n x K) <= softmax( (B x n x K) )
        residual_softmax_factors = tf.nn.softmax(smoothed_sqnorms)

        # (B x n x K x 1) <= (B x n x K)
        residual_softmax_factors = tf.expand_dims(residual_softmax_factors, axis=-1)

        # (B x n x K x C) <= (B x n x K x C) * (B x n x K x 1)
        scaled_residuals = residuals * residual_softmax_factors

        # Sum over "spacial" dimensions
        # (B x K x C) <= reduce_sum( (B x n x K x C) )
        codeword_encodings = tf.reduce_sum(scaled_residuals, axis=1)
        codeword_encodings_batch_norm = self.batch_norm(codeword_encodings)
        codeword_encodings_batch_norm_relu = self.relu(codeword_encodings_batch_norm)

        # (B x C) <= reduce_sum( (B x K x C) )
        output_encodings = tf.reduce_sum(codeword_encodings_batch_norm_relu, axis=1)

        return output_encodings
