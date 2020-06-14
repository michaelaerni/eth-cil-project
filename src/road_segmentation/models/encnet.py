import tensorflow as tf

"""
Implementation of encoder layers to build EncNets according to (http://arxiv.org/abs/1803.08904)
"""


class ContextEncodingModule(tf.keras.layers.Layer):
    """
    The full context encoding module as in the paper. It takes as input some standard tensor and outputs a tuple of
    tensors. The first one is the input where the feature maps are scaled by some attention vector computed by the
    encoder and the second tensor is what the semantic encoding loss is applied to.
    """

    _INITIALIZER = 'he_uniform'
    """
    Initializer for dense layer weights.
    """

    def __init__(
            self,
            codewords: int,
            **kwargs
    ):
        """
        Args:
            codewords: Number of codewords to be used.
        """
        super(ContextEncodingModule, self).__init__(**kwargs)

        self.encoder = Encoder(codewords)

        # Features will be set in build.
        self.fully_connected_encoding = tf.keras.layers.Dense(
            0,
            activation='sigmoid',
            kernel_initializer=self._INITIALIZER,
        )

        # No activation for se loss output. The output dimension is 1 because in our setting we only have to detect the
        # presence of road. Thus, our objective is somewhat different to the original paper, where they detect presence
        # of multiple classes.
        self.fully_connected_se_loss = tf.keras.layers.Dense(
            1,
            activation=None,
            kernel_initializer=self._INITIALIZER
        )

    def build(self, input_shape):
        features = input_shape[-1]
        self.fully_connected_encoding.units = features

    def call(self, inputs, **kwargs):
        encodings = self.encoder(inputs)

        featuremaps_attention = self.fully_connected_encoding(encodings)
        featuremaps_attention = tf.expand_dims(tf.expand_dims(featuremaps_attention, axis=1), axis=1)

        featuremaps_output = featuremaps_attention * inputs

        se_loss_output = self.fully_connected_se_loss(encodings)

        return featuremaps_output, se_loss_output


class Encoder(tf.keras.layers.Layer):
    """
    Encoder as described in the paper. The output is a single vector (per batch element).
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

        self.n_codewords = codewords

        self.codewords = None

        # The smoothing factors can not become negative. This deviates from the original implementation but is more
        # reasonable, as otherwise the negative factor in the softmax is pointless.
        self.smoothing_factors = self.add_weight(
            name='smoothing_factors',
            shape=(1, 1, self.n_codewords),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=0,
                maxval=1
            ),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def build(self, input_shape):
        features = input_shape[-1]

        # Initialize codewords variables, according to implementation of main author at
        # https://github.com/zhanghang1989/PyTorch-Encoding/blob/d9dea1724e38362a7c75ca9498f595248f283f00/encoding/nn/encoding.py#L86
        support = 1. / ((self.n_codewords * features) ** (1 / 2))
        self.codewords = self.add_weight(
            name='codewords',
            shape=(1, 1, self.n_codewords, features),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=-support,
                maxval=support
            ),
            trainable=True
        )

    def call(self, inputs, **kwargs):
        # Assuming that the input shape is (B x H x W x C), where B = batch size, W = width, H = height, K = number of
        # codewords and C = number of channels/features.
        inputs_shape = tf.shape(inputs)
        batch_size, height, width, num_features = tf.unstack(inputs_shape)

        # Number of "pixels" = height * width
        n = height * width

        # (B x n x 1 x C) <= (B x H x W x C)
        features = tf.reshape(inputs, shape=(batch_size, n, 1, num_features))

        # (B x n x K x C) <= (B x n x 1 x C) - (1 x 1 x K x C)
        # Pairwise differences between codewords and features
        # FIXME: Most likely possible to compute this more efficiently with squared norm.
        residuals = features - self.codewords

        # Squared norm.
        # (B x n x K) <= (B x n x K x C)
        residual_sqnorms = tf.reduce_sum(tf.square(residuals), axis=-1)

        # Calculate the un-normalized pairwise residual weights
        # (B x n x K) <= (B x n x K) * (1 x 1 x K)
        smoothed_sqnorms = residual_sqnorms * self.smoothing_factors

        # (B x n x K) <= (B x n x K), (B x n) distributions
        residual_softmax_factors = tf.nn.softmax(-smoothed_sqnorms, axis=-1)

        # (B x n x K x 1) <= (B x n x K)
        residual_softmax_factors = tf.expand_dims(residual_softmax_factors, axis=-1)

        # (B x n x K x C) <= (B x n x K x C) * (B x n x K x 1)
        scaled_residuals = residuals * residual_softmax_factors

        # Sum over "spatial" dimensions.
        # (B x K x C) <= (B x n x K x C)
        codeword_encodings = tf.reduce_sum(scaled_residuals, axis=1)
        codeword_encodings_batch_norm = self.batch_norm(codeword_encodings)

        # FIXME: Why ReLU? Intuitively, this should decrease the performance.
        codeword_encodings_batch_norm_relu = self.relu(codeword_encodings_batch_norm)

        # (B x C) <= (B x K x C)
        output_encodings = tf.reduce_sum(codeword_encodings_batch_norm_relu, axis=1)

        return output_encodings
