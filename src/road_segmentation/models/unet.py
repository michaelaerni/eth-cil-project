import tensorflow as tf

"""
Implementation of plain unet (https://arxiv.org/abs/1505.04597).
"""


class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()

        self.contracting_path = [
            conv_block(64, 3, 1, name="down_block_1"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_1"),
            conv_block(128, 3, 1, name="down_block_2"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_2"),
            conv_block(256, 3, 1, name="down_block_3"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_3"),
            conv_block(512, 3, 1, dropout_rate=0.5, name="down_block_4"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_4"),
            conv_block(1024, 3, 1, name="lowest_block"),
        ]
        self.expansive_path = [
            upsample(512, 3, name="up_conv_1"),
            conv_block(512, 3, 1, name="conv_block_right_1"),
            upsample(256, 3, name="up_conv_2"),
            conv_block(256, 3, 1, name="conv_block_right_2"),
            upsample(128, 3, name="up_conv_3"),
            conv_block(128, 3, 1, name="conv_block_right_3"),
            upsample(64, 3, name="up_conv_4"),
            conv_block(64, 3, 1, name="conv_block_right_4"),
        ]

        self.conv_out = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same', activation=None)

    def call(self, inputs, training=None, mask=None):
        # inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = inputs

        # Downsampling and saving outputs to establish skip connection
        skips = []
        for block in self.contracting_path:
            x = block(x)
            if "max_pool" not in block.name and len(skips) < 4:
                skips.append(x)
        skips = list(reversed(skips))

        # Upsampling, cropping and concatenation with skip connections
        counter = 0
        for block in self.expansive_path:
            x = block(x)
            if "up_conv" in block.name and counter < 4:
                concat = tf.keras.layers.Concatenate()
                x = concat([x, crop_to_fit(x, skips[counter])])
                counter += 1

        # convert to output
        logits = self.conv_out(x)

        return logits


def conv_block(filters: int, size, stride=(1, 1), use_batch_norm: bool = False, dropout_rate=None,
               name: str = None) -> tf.keras.Sequential:
    """
    Conv2D => (Dropout) => ReLu => Conv2D => (Dropout) => ReLu

    :param filters: number of filters
    :param size: size of ilters
    :param stride: stride for conv2D
    :param use_batch_norm: if True batch normalization is applied after each conv layer
    :param dropout_rate: either None (no dropout) or float between 0 and 1 (apply dropout)
    :param name: optional a name for the conv block
    :return: conv block as sequential model
    """
    result = tf.keras.Sequential(name=name)
    for i in range(2):
        result.add(tf.keras.layers.Conv2D(filters, size, strides=stride, padding='valid',
                                          kernel_initializer='he_normal', use_bias=False))

        if use_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        if dropout_rate:
            result.add(tf.keras.layers.Dropout(dropout_rate))

        result.add(tf.keras.layers.ReLU())

    return result


def upsample(filters, size, stride=(2, 2), use_batch_norm=None, name=None) -> tf.keras.Sequential:
    """
    Conv2DTranspose => (BatchNorm) => ReLu

    :param filters: number of filters
    :param size: size of filters
    :param stride: stride for conv2D
    :param use_batch_norm: if True batch normalization is applied after each conv layer
    :param name: optional a name for the conv block
    :return: up sample block as sequential model
    """
    result = tf.keras.Sequential(name=name)
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=stride,
                                        padding='same',
                                        kernel_initializer='he_normal',
                                        use_bias=False))
    if use_batch_norm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def crop_to_fit(main, to_crop):
    return tf.image.resize_with_crop_or_pad(to_crop, tf.shape(main)[1], tf.shape(main)[2])
