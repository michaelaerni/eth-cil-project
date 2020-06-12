import tensorflow as tf


class JPUModule(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    # TODO: Compare with FastFCN implementation

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

        # TODO: I think the output uses a 1x1 convolution, but am not quite sure about what the original author does
        # TODO: Also, I think convolution commutates here and this is equivalent to a 1x1 512 conv, but also check this
        # TODO: I now think the above sentence is wrong
        #output = tf.concat(dilation_outputs, axis=-1)
        dilation_outputs = tf.stack(dilation_outputs, axis=0)
        output = tf.reduce_sum(dilation_outputs, axis=0)
        return output


class JPUInputBlock(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            features: int,
            kernel_initializer: str,
            weight_decay: float,
            **kwargs
    ):
        super(JPUInputBlock, self).__init__(**kwargs)

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
            kernel_initializer: str,
            weight_decay: float,
            **kwargs
    ):
        super(JPUSeparableBlock, self).__init__(**kwargs)

        # Convolution on inputs with dilation
        self.conv_in = tf.keras.layers.Conv2D(
            filters=3 * features,
            kernel_size=3,
            padding='same',
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_in = tf.keras.layers.BatchNormalization()

        # 1x1 convolution for output (no dilation)
        self.conv_pointwise = tf.keras.layers.Conv2D(
            filters=features,
            kernel_size=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm_pointwise = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        features_in = self.conv_in(inputs)
        features_in = self.batch_norm_in(features_in)

        features_out = self.conv_pointwise(features_in)
        features_out = self.batch_norm_pointwise(features_out)
        output = self.activation(features_out)

        return output


class FCNHead(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            intermediate_features: int,
            kernel_initializer: str,
            dropout_rate: float = 0.1,
            weight_decay: float = 1e-4,
            **kwargs
    ):
        super(FCNHead, self).__init__(**kwargs)

        # Normal conv -> batch norm -> relu
        self.conv_in = tf.keras.layers.Conv2D(
            filters=intermediate_features,
            kernel_size=3,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

        # Dropout before output
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Output (without activation or anything)
        self.conv_out = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, inputs, **kwargs):
        intermeditate_features = self.conv_in(inputs)
        intermeditate_features = self.batch_norm(intermeditate_features)
        intermeditate_features = self.activation(intermeditate_features)
        intermeditate_features = self.dropout(intermeditate_features)

        output_features = self.conv_out(intermeditate_features)
        return output_features
