import typing

import tensorflow as tf

"""
Implementation of layers to build EncNets according to (http://arxiv.org/abs/1803.08904)
"""


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