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
            features=features,
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
            features: int,
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

        self.features = features
        self.encoder = Encoder(codewords, features)
        self.fully_connected_encoding = tf.keras.layers.Dense(
            features,
            activation='sigmoid'
        )
        self.fully_connected_SE_loss = tf.keras.layers.Dense(
            classes,
            activation='sigmoid'
        )

    def call(self, inputs, **kwargs):
        encodings = self.encoder(inputs)
        featuremaps_attention = self.fully_connected_encoding(encodings)

        # TODO: Check if shapes match: This operation works if "featuremaps_attention" is of 3 dimensions and inputs
        #       has 4 dimensions, but is only correct if "featuremaps_attention" is 4 dimensions as well!
        featuremaps_output = featuremaps_attention * inputs

        encodings = tf.reshape(encodings, (tf.shape(encodings)[0], self.features))
        SE_loss_output = self.fully_connected_SE_loss(encodings)

        return featuremaps_output, SE_loss_output


class Encoder(tf.keras.layers.Layer):
    """
    TODO: Documentation
    FIXME: Refactor possibly into multiple sub classes, what makes sense is TBD.
    """

    def __init__(
            self,
            codewords: int,
            features: int
    ):
        """
        Args:
            codewords: Number of codewords to be used.
            features: Dimension (length) of codewords.
        """
        super(Encoder, self).__init__()

        # FIXME: We need the number of codewords in the call function, which results in a name conflict of the
        #  codewords tensor with the actual number of codewords. Hence, breaking the naming convention here.
        self.n_codewords = codewords
        self.features = features

        # Initialize codewords variables, according to implementation of main author at
        # https://github.com/zhanghang1989/PyTorch-Encoding/blob/d9dea1724e38362a7c75ca9498f595248f283f00/encoding/nn/encoding.py#L86
        standard_dev = 1. / ((codewords * features) ** (1 / 2))

        # FIXME: Is this the correct way to initialize variables?
        codewords_initializer = tf.random_uniform_initializer(
            minval=-standard_dev,
            maxval=standard_dev
        )
        codewords_initial_values = codewords_initializer((features, codewords))

        smoothing_factors_initializer = tf.random_uniform_initializer(
            minval=-1,
            maxval=0
        )
        smoothing_factors_initial_values = smoothing_factors_initializer([codewords])

        # FIXME: Will vars declared this way be updated as expected?
        self.codewords = tf.keras.backend.variable(
            codewords_initial_values,
            dtype=tf.float32,
            name='codewords'
        )

        self.smoothing_factors = tf.keras.backend.variable(
            smoothing_factors_initial_values,
            dtype=tf.float32,
            name='smoothing_factors'
        )

        # Want to apply batch norm on the channels axis, not on the codewords axis
        self.batch_norm = tf.keras.layers.BatchNormalization(axis=-2)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        # TODO: This code needs to be tested thoroughly!
        # Assuming that the input shape is (Batch * width * height * features)
        in_shape = tf.shape(inputs)

        WH = in_shape[1] * in_shape[2]

        features_shape = (in_shape[0], WH, self.features, 1)
        # (B x W x H x C) => (B x W*H x C x 1)
        expanded_features = tf.reshape(inputs, shape=features_shape)
        # (B x W*H x C x 1) => (B x W*H x C x K)
        expanded_features = tf.repeat(expanded_features, self.n_codewords, axis=-1)

        # (C x K) => (1 x C x K)
        expanded_codewords = tf.expand_dims(self.codewords, axis=0)
        # (1 x C x K) => (W*H x C x K)
        expanded_codewords = tf.repeat(expanded_codewords, features_shape[1], axis=0)

        # (B x W*H x C x K) = (B x W*H x C x K) - (W*H x C x K)
        residuals = expanded_features - expanded_codewords

        # (B x W*H x K)
        residual_sqnorms = tf.square(tf.norm(residuals, axis=-2))

        # (K) => (1 x K)
        expanded_smoothing_factors = tf.expand_dims(self.smoothing_factors, axis=0)
        # (1 x K) => (W*H x K)
        expanded_smoothing_factors = tf.repeat(expanded_smoothing_factors, WH, axis=0)

        # (B x WH x K) = (B x WH x K) * (WH x K)
        neg_smoothed_sqnorms = - (residual_sqnorms * expanded_smoothing_factors)

        # (B x WH x K) = softmax( (B x WH x K) )
        residual_softmax_factors = tf.nn.softmax(neg_smoothed_sqnorms)

        # (B x WH x K) => (B x WH x 1 x K)
        residual_softmax_factors = tf.expand_dims(residual_softmax_factors, axis=-2)

        # (B x WH x C x K) = (B x WH x C x K) * (B x WH x 1 x K)
        scaled_residuals = residuals * residual_softmax_factors

        # Sum over "spacial" dimensions
        # (B x C x K) = reduce_sum( (B x WH x C x K) )
        codeword_encodings = tf.reduce_sum(scaled_residuals, axis=1)
        codeword_encodings_batch_norm = self.batch_norm(codeword_encodings)
        codeword_encodings_batch_norm_relu = self.relu(codeword_encodings_batch_norm)

        # (B x C) = reduce_sum( (B x C x K) )
        output_encodings = tf.reduce_sum(codeword_encodings_batch_norm_relu, axis=-1)

        # (B x C) => (B x 1 x 1 x C)
        output_encodings = tf.reshape(output_encodings, (in_shape[0], 1, 1, self.features))

        return output_encodings


class EncNetLoss(tf.keras.losses.Loss):
    """
    Custom loss to deal with tuple output of classifier
    """

    # FIXME: Encoding loss is meant to learn presence of classes, which does not really make
    #  sense in our context, where we only have one class and they are generally both present.
    def __init__(self):
        super(EncNetLoss, self).__init__()

        self.loss_seg = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.loss_SE = tf.keras.losses.BinaryCrossentropy()

    def call(self, true_seg, prediction):
        pred_seg, pred_SE = prediction

        pred_shape = tf.shape(true_seg)
        true_SE = tf.map_fn(lambda seg: tf.reduce_sum(seg), true_seg) / (pred_shape[1] * pred_shape[2])

        out_loss_SE = self.loss_SE(true_SE, pred_SE)
        out_loss_seg = self.loss_seg(true_seg, pred_seg)

        return out_loss_SE + out_loss_seg
