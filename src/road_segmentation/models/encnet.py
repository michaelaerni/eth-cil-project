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

    _CLASSES = 1
    """
    In our case we only have "road" or "not road", therefore the global context output is of size one.
    """

    def __init__(
            self,
            codewords: int,
            features: int,
            **kwargs
    ):
        """
        Args:
            codewords: Number of codewords to be used.
            features: Number of features.
            classes: The number of classes in the segmentation task.
        """
        super(ContextEncodingModule, self).__init__(**kwargs)

        self.encoder = Encoder(codewords)

        # This is pytorch default for weights and biases, which is what the original implementation uses.
        init_support = tf.sqrt(1. / features)
        initializer = tf.random_uniform_initializer(
            minval=-init_support,
            maxval=init_support
        )

        self.fully_connected_encoding = tf.keras.layers.Dense(
            features,
            activation='sigmoid',
            kernel_initializer=initializer,
            bias_initializer=initializer
        )

        # No activation for se loss output
        self.fully_connected_se_loss = tf.keras.layers.Dense(
            self._CLASSES,
            activation=None,
            kernel_initializer=initializer,
            bias_initializer=initializer
        )

    def call(self, inputs, **kwargs):
        encodings = self.encoder(inputs)
        featuremaps_attention = self.fully_connected_encoding(encodings)

        featuremaps_attention_shape = tf.shape(featuremaps_attention)
        newshape = (featuremaps_attention_shape[0], 1, 1, featuremaps_attention_shape[1])
        featuremaps_attention = tf.reshape(featuremaps_attention, newshape)
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

        self.smoothing_factors = self.add_weight(
            name='smoothing_factors',
            shape=(1, 1, self.n_codewords),
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(
                minval=-1,
                maxval=0
            ),
            trainable=True
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
        batch_size, height, width, num_features = inputs_shape[0], inputs_shape[1], inputs_shape[2], inputs_shape[3]

        # Number of "pixels" = width * height
        n = height * width

        # (B x n x 1 x C) <= (B x H x W x C)
        features = tf.reshape(inputs, shape=(batch_size, n, 1, num_features))

        # (B x n x K x C) <= (B x n x 1 x C) - (1 x 1 x K x C)
        # Pairwise differences between codewords and features
        residuals = features - self.codewords

        # Squared norm.
        # (B x n x K) <= (B x n x K x C)
        residual_sqnorms = tf.reduce_sum(tf.square(residuals), axis=-1)

        # FIXME: Compared to the paper, there is a minus missing.
        #  This is compensated by initializing the smoothing weights with a negative value.
        #  Conceptually, enforcing positive smoothing weights (and a minus sign) makes more sense.
        # Calculate the un-normalized pairwise residual weights
        # (B x n x K) <= (B x n x K) * (1 x 1 x K)
        smoothed_sqnorms = residual_sqnorms * self.smoothing_factors

        # (B x n x K) <= (B x n x K), (B x n) distributions
        residual_softmax_factors = tf.nn.softmax(smoothed_sqnorms, axis=-1)

        # (B x n x K x 1) <= (B x n x K)
        residual_softmax_factors = tf.expand_dims(residual_softmax_factors, axis=-1)

        # (B x n x K x C) <= (B x n x K x C) * (B x n x K x 1)
        scaled_residuals = residuals * residual_softmax_factors

        # Sum over "spacial" dimensions.
        # (B x K x C) <= (B x n x K x C)
        codeword_encodings = tf.reduce_sum(scaled_residuals, axis=1)
        codeword_encodings_batch_norm = self.batch_norm(codeword_encodings)
        codeword_encodings_batch_norm_relu = self.relu(codeword_encodings_batch_norm)

        # (B x C) <= (B x K x C)
        output_encodings = tf.reduce_sum(codeword_encodings_batch_norm_relu, axis=1)

        return output_encodings
