import typing

import tensorflow as tf


class TestFastFCN(tf.keras.models.Model):
    """
    FIXME: This is just a test class and should be renamed and moved
    """

    KERNEL_INITIALIZER = 'he_normal'  # FIXME: This is somewhat arbitrarily chosen

    def __init__(
            self,
            jpu_features: int,
            weight_decay: float,
            output_upsampling: str
    ):
        super(TestFastFCN, self).__init__()

        self.backbone = rs.models.resnet.ResNet50Backbone(weight_decay=weight_decay)
        self.upsampling = rs.models.fastfcn.JPUModule(
            features=jpu_features,
            weight_decay=weight_decay
        )

        # FIXME: Head is only for testing, replace this with EncNet head
        self.head = rs.models.fastfcn.FCNHead(
            intermediate_features=256,
            kernel_initializer=self.KERNEL_INITIALIZER,
            weight_decay=weight_decay
        )

        # FIXME: Upsampling of the 8x8 output is slightly unnecessary and should be done more in line with the s16 target
        self.output_upsampling = tf.keras.layers.UpSampling2D(size=(8, 8), interpolation=output_upsampling)

        # FIXME: The paper uses an auxiliary FCNHead at the end to calculate the loss, but never for the output...
        #  Does not really make sense and is also not mentioned in the paper I think

    def call(self, inputs, training=None, mask=None):
        _, input_height, input_width, _ = tf.unstack(tf.shape(inputs))
        padded_inputs = rs.util.pad_to_stride(inputs, target_stride=32, mode='REFLECT')

        intermediate_features = self.backbone(padded_inputs)[-3:]

        upsampled_features = self.upsampling(intermediate_features)

        small_outputs = self.head(upsampled_features)

        padded_outputs = self.output_upsampling(small_outputs)
        outputs = tf.image.resize_with_crop_or_pad(padded_outputs, input_height, input_width)

        return outputs


class JPUModule(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    _KERNEL_INITIALIZER = 'he_normal'  # TODO: Which initializer is actually used?
    _INTERPOLATION = 'bilinear'
    _DILATION_RATES = (1, 2, 4, 8)

    def __init__(
            self,
            features: int = 512,
            weight_decay: float = 1e-4,
            **kwargs
    ):
        super(JPUModule, self).__init__(**kwargs)

        # Per-resolution convolution blocks
        self.initial_s32 = JPUInputBlock(
            features,
            self._KERNEL_INITIALIZER,
            weight_decay
        )
        self.initial_s16 = JPUInputBlock(
            features,
            self._KERNEL_INITIALIZER,
            weight_decay
        )
        self.initial_s8 = JPUInputBlock(
            features,
            self._KERNEL_INITIALIZER,
            weight_decay
        )

        # Upsampling from stride 32 to 16 and 16 to 8
        self.upsampling_s32 = tf.keras.layers.UpSampling2D(
            size=(4, 4),
            interpolation=self._INTERPOLATION
        )
        self.upsampling_s16 = tf.keras.layers.UpSampling2D(
            size=(2, 2),
            interpolation=self._INTERPOLATION
        )

        # Parallel dilated convolutions
        self.separable_blocks = [
            JPUSeparableBlock(features, dilation_rate, self._KERNEL_INITIALIZER, weight_decay)
            for dilation_rate in self._DILATION_RATES
        ]

    def call(self, inputs, **kwargs):
        inputs_s8, inputs_s16, inputs_s32 = inputs

        # Per-resolution convolutions
        features_s8 = self.initial_s8(inputs_s8)
        features_s16 = self.initial_s16(inputs_s16)
        features_s32 = self.initial_s32(inputs_s32)

        # Upsample and concatenate
        upsampled_s16 = self.upsampling_s16(features_s16)
        upsampled_s32 = self.upsampling_s32(features_s32)
        dilation_inputs = tf.concat([features_s8, upsampled_s16, upsampled_s32], axis=-1)

        # Parallel dilated convolutions
        dilation_outputs = [block(dilation_inputs) for block in self.separable_blocks]

        # The paper proposes to perform a 1x1 convolution here.
        # The reference implementation does that directly in the heads.
        # FIXME: When merging with the Encnet head keep this in mind!
        output = tf.concat(dilation_outputs, axis=-1)
        return output


class JPUInputBlock(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            features: int,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            weight_decay: float,
            **kwargs
    ):
        super(JPUInputBlock, self).__init__(**kwargs)

        # Bias in the convolution layer is omitted since the batch normalization adds a bias term itself
        self.conv = tf.keras.layers.Conv2D(
            filters=features,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        features = self.conv(inputs)
        features = self.batch_norm(features)
        output = self.activation(features)
        return output


class JPUSeparableBlock(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            features: int,
            dilation_rate: int,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            weight_decay: float,
            **kwargs
    ):
        super(JPUSeparableBlock, self).__init__(**kwargs)

        # Compared to the original implementation, this only performs batch norm once at the end

        # Bias is omitted since the batch normalization adds a bias term itself
        self.conv = tf.keras.layers.SeparableConv2D(
            features,
            kernel_size=3,
            padding='same',
            dilation_rate=dilation_rate,
            depth_multiplier=1,
            activation=None,
            use_bias=False,
            depthwise_initializer=kernel_initializer,
            pointwise_initializer=kernel_initializer,
            depthwise_regularizer=tf.keras.regularizers.l2(weight_decay),
            pointwise_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        features = self.conv(inputs)
        features = self.batch_norm(features)
        output = self.activation(features)
        return output


class FCNHead(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            intermediate_features: int,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            dropout_rate: float = 0.1,
            weight_decay: float = 1e-4,
            **kwargs
    ):
        super(FCNHead, self).__init__(**kwargs)

        # Input 1x1 convolution
        # The original paper proposes this as part of the JPU but the reference implementation does not.
        # It is performed here primarily for performance and memory reasons.
        # Bias in the convolution layer is omitted since the batch normalization adds a bias term itself
        self.conv_in = tf.keras.layers.Conv2D(
            filters=intermediate_features,
            kernel_size=1,
            padding='valid',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_in = tf.keras.layers.BatchNormalization()
        self.activation_in = tf.keras.layers.ReLU()

        # Normal conv -> batch norm -> relu
        # Bias in the convolution layer is omitted since the batch normalization adds a bias term itself
        self.conv_middle = tf.keras.layers.Conv2D(
            filters=intermediate_features,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_middle = tf.keras.layers.BatchNormalization()
        self.activation_middle = tf.keras.layers.ReLU()

        # Dropout before output
        self.dropout = tf.keras.layers.SpatialDropout2D(dropout_rate)

        # Output (without activation or anything)
        self.conv_out = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='valid',
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, inputs, **kwargs):
        compressed_features = self.conv_in(inputs)
        compressed_features = self.batch_norm_in(compressed_features)
        compressed_features = self.activation_in(compressed_features)

        intermediate_features = self.conv_middle(compressed_features)
        intermediate_features = self.batch_norm_middle(intermediate_features)
        intermediate_features = self.activation_middle(intermediate_features)
        intermediate_features = self.dropout(intermediate_features)

        output_features = self.conv_out(intermediate_features)
        return output_features
