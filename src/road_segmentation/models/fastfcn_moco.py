import typing

import tensorflow as tf

import road_segmentation as rs

from road_segmentation.models.fastfcn import FastFCN, FCNHead, EncoderHead


class FastFCNMoCoContrastBackbone(tf.keras.Model):

    def __init__(
            self,
            fastfcn_moco_backbone,
            **kwargs
    ):
        super(FastFCNMoCoContrastBackbone, self).__init__(**kwargs)

        self.fastfcn_moco_backbone = fastfcn_moco_backbone

    def call(self, inputs, **kwargs):
        _, encodings = self.fastfcn_moco_backbone(inputs)
        return encodings


class EncodingsDense(tf.keras.layers.Dense):
    def __init__(
            self,
            **kwargs
    ):
        kwargs['units'] = 0
        super(EncodingsDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.units = input_shape[-1]
        super(EncodingsDense, self).build(input_shape)


class FastFCNMoCo(tf.keras.Model):
    def __init__(
            self,
            fastfcn_moco_backbone,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            output_upsampling: str,
            dropout_rate: float = 0.1,
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super(FastFCNMoCo, self).__init__(**kwargs)
        self.fastfcn_moco_backbone = fastfcn_moco_backbone

        self.fully_connected_encoding = EncodingsDense(
            activation='sigmoid',
            kernel_initializer=dense_initializer
        )

        self.fully_connected_se_loss = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            kernel_initializer=dense_initializer
        )

        self.dropout = tf.keras.layers.SpatialDropout2D(dropout_rate)
        self.conv_out = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='valid',
            activation=None,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        self.output_upsampling = tf.keras.layers.UpSampling2D(
            size=(8, 8),
            interpolation=output_upsampling
        )

    def call(self, inputs, **kwargs):
        _, input_height, input_width, _ = tf.unstack(tf.shape(inputs))

        encoder_features_inputs, encodings = self.fastfcn_moco_backbone(inputs)

        featuremaps_attention = self.fully_connected_encoding(encodings)
        weighted_features = encoder_features_inputs * featuremaps_attention

        se_loss_features = tf.squeeze(self.fully_connected_se_loss(encodings), axis=[1, 2])

        pre_output_features = self.dropout(weighted_features)
        small_output_features = self.conv_out(pre_output_features)

        padded_outputs = self.output_upsampling(small_output_features)
        outputs = tf.image.resize_with_crop_or_pad(padded_outputs, input_height, input_width)
        return outputs, se_loss_features


class FastFCNMoCoBackbone(tf.keras.Model):
    """
    FastFCN Backbone with context encoding module outputs as output.
    """

    _INTERMEDIATE_FEATURES = 512

    def __init__(
            self,
            resnet_backbone,
            jpu_features: int,
            kernel_initializer: typing.Union[str, tf.keras.initializers.Initializer],
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super(FastFCNMoCoBackbone, self).__init__(**kwargs)

        self.resnet_backbone = resnet_backbone
        self.upsampling = rs.models.fastfcn.JPUModule(
            features=jpu_features,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        self.conv_in = tf.keras.layers.Conv2D(
            filters=self._INTERMEDIATE_FEATURES,
            kernel_size=1,
            padding='valid',
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer
        )

        self.batch_norm_in = tf.keras.layers.BatchNormalization()
        self.activation_in = tf.keras.layers.ReLU()

        self.encoder = rs.models.encnet.Encoder(
            codewords=32
        )

    def call(self, inputs, **kwargs):
        _, input_height, input_width, _ = tf.unstack(tf.shape(inputs))
        padded_inputs = rs.util.pad_to_stride(inputs, target_stride=32, mode='REFLECT')

        intermediate_features = self.resnet_backbone(padded_inputs)[-3:]
        upsampled_features = self.upsampling(intermediate_features)

        compressed_features = self.conv_in(upsampled_features)
        compressed_features = self.batch_norm_in(compressed_features)
        compressed_features = self.activation_in(compressed_features)
        encodings = self.encoder(compressed_features)
        encodings = tf.expand_dims(tf.expand_dims(encodings, axis=1), axis=1)

        return compressed_features, encodings
