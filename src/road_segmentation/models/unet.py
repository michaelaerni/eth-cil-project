import typing

import tensorflow as tf


class UNet(tf.keras.Model):
    """
    Implementation of plain U-Net according to original paper (https://arxiv.org/abs/1505.04597).
    """

    _DEFAULT_FILTERS = [64, 128, 256, 512]
    """
    Default number of filters according to the original paper.
    """

    _DEFAULT_BOTTLENECK_FILTERS = 1024
    """
    Default number of filters for the bottleneck layer according to the original paper.
    """

    def __init__(
        self,
        filters: typing.Optional[typing.List[int]] = None,
        bottleneck_filters: typing.Optional[int] = None,
        input_padding: typing.Tuple[
            typing.Tuple[int, int],
            typing.Tuple[int, int]
        ] = ((0, 0), (0, 0)),
        apply_batch_norm: bool = False,
        dropout_rate: typing.Optional[float] = None
    ):
        """
        U-Net fully convolutional segmentation network.

        Args:
            filters:
                List of filters per spatial dimension.
                The number of entries n in this list determine the downsampling factor (2^n).
                Defaults to the number of filters in the original paper.
            bottleneck_filters:
                Number of filters to use in the bottleneck layer.
                Defaults to the number of filters in the original paper.
            input_padding: Input padding in the format ((top, bottom), (left, right)).
            apply_batch_norm: If true then batch normalization will be applied prior to activations in conv layers.
            dropout_rate: If not None specifies the dropout rate in [0, 1] for the final features of each block.
        """

        super(UNet, self).__init__()

        # Use default numbers of filters if nothing is specified
        if filters is None:
            filters = self._DEFAULT_FILTERS
        if bottleneck_filters is None:
            bottleneck_filters = self._DEFAULT_BOTTLENECK_FILTERS

        # Padding will be applied in call
        self.input_padding = input_padding

        # Contracting path until bottleneck, store respective blocks and pooling together
        self.contracting_path = [(
                UNetConvBlock(current_filters, apply_batch_norm, dropout_rate),
                tf.keras.layers.MaxPool2D(pool_size=(2, 2))
            )
            for current_filters in filters
        ]

        # Bottleneck
        self.bottleneck = UNetConvBlock(bottleneck_filters, apply_batch_norm, dropout_rate)

        # Expanding path, store again upsampling and blocks together
        self.expanding_path = [(
                UNetConvBlock(current_filters, apply_batch_norm, dropout_rate),
                UNetUpsampleBlock(current_filters, apply_batch_norm, dropout_rate)
            )
            for current_filters in reversed(filters)
        ]

        # Output convolution
        self.output_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='valid',
            activation=None
        )

        # Output will be cropped in call

    def call(self, inputs, training=None, mask=None):
        # Pad inputs
        input_shape = tf.shape(inputs)
        features = tf.pad(
            inputs,
            ((0, 0),) + self.input_padding + ((0, 0),),  # Do not pad batch or features
            mode='REFLECT'
        )

        # Contracting path, storing features for skip connections
        intermediate_features = []
        for current_block, current_downsampling in self.contracting_path:
            # Apply block
            current_features = current_block(features)

            # Store features for skip connections
            intermediate_features.append(current_features)

            # Downsample
            features = current_downsampling(current_features)

        # Bottleneck
        features = self.bottleneck(features)

        # Expanding path, applying skip connections
        skip_connection_features = reversed(intermediate_features)
        for (current_block, current_upsampling), skip_features in zip(self.expanding_path, skip_connection_features):
            # Upsample
            upsampled_features = current_upsampling(features)

            # Crop and concatenate features from skip connection
            print(tf.shape(upsampled_features)[1], tf.shape(upsampled_features)[2])
            combined_features = tf.concat([
                tf.image.resize_with_crop_or_pad(
                    skip_features,
                    target_height=tf.shape(upsampled_features)[1],
                    target_width=tf.shape(upsampled_features)[2]
                ),
                upsampled_features
            ],  axis=-1)

            # Apply block
            features = current_block(combined_features)

        # Apply output convolution
        output_logits = self.output_conv(features)

        # Crop output to same shape as input (will be larger)
        cropped_logits = tf.image.resize_with_crop_or_pad(
            output_logits,
            target_height=input_shape[1],
            target_width=input_shape[2]
        )

        return cropped_logits


class UNetConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        apply_batch_norm: bool = False,
        dropout_rate: typing.Optional[float] = None,
        **kwargs
    ):
        """
        U-Net convolution block for a fixed spatial size.

        This performs Conv2D => (BN) => ReLu => Conv2D => (BN) => ReLu => (Dropout)
        where terms in braces are optional.

        Args:
            filters: Number of output filters.
            apply_batch_norm: If true then batch normalization will be applied prior to activations.
            dropout_rate: If not None specifies the dropout rate in [0, 1] for the final features.
        """
        super(UNetConvBlock, self).__init__(**kwargs)

        # Create convolution layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='valid',
            kernel_initializer='he_normal',
            use_bias=True
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters,
            kernel_size=(3, 3),
            padding='valid',
            kernel_initializer='he_normal',
            use_bias=True
        )

        # Create activation layers
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()

        # Create optional batch normalisation layers
        self.batch_norm1 = tf.keras.layers.BatchNormalization() if apply_batch_norm else None
        self.batch_norm2 = tf.keras.layers.BatchNormalization() if apply_batch_norm else None

        # Create optional droput
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate is not None else None

    def call(self, inputs, **kwargs):
        features = self.conv1(inputs)
        if self.batch_norm1 is not None:
            features = self.batch_norm1(features)
        activations = self.relu1(features)

        features = self.conv2(activations)
        if self.batch_norm2 is not None:
            features = self.batch_norm2(features)
        outputs = self.relu2(features)

        # Apply optional dropout
        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return super(UNetConvBlock, self).call(outputs, **kwargs)


class UNetUpsampleBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        apply_batch_norm: bool = False,
        dropout_rate: typing.Optional[float] = None,
        **kwargs
    ):
        """
        U-Net upsampling block which increases the spatial dimensions by a factor of 2.

        Args:
            filters: Number of output filters.
            apply_batch_norm: If true then batch normalization will be applied prior to activations.
            dropout_rate: If not None specifies the dropout rate in [0, 1] for the final features.
        """
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            kernel_initializer='he_normal',
            use_bias=False
        )

        # Optional steps
        self.batch_norm = tf.keras.layers.BatchNormalization() if apply_batch_norm else None
        self.dropout = tf.keras.layers.Dropout(dropout_rate) if dropout_rate is not None else None

    def call(self, inputs, **kwargs):
        outputs = self.conv(inputs)

        # Optionally apply batch normalization and/or dropout
        if self.batch_norm is not None:
            outputs = self.batch_norm(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)

        # No activation is performed!
        return outputs
