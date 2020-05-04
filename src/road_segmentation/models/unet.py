import tensorflow as tf


class UNet(tf.keras.Model):
    """
    Implementation of plain U-Net according to original paper (https://arxiv.org/abs/1505.04597).
    """

    def __init__(self, dropout_rate: float, apply_dropout_after_conv_blocks: bool, upsampling_method: str):
        super(UNet, self).__init__()
        after_conv_block_dropout_rate = None
        if apply_dropout_after_conv_blocks:
            after_conv_block_dropout_rate = dropout_rate
        self.contracting_path = [
            conv_block(filters=64,
                       size=3,
                       stride=1,
                       dropout_rate=after_conv_block_dropout_rate,
                       name="down_block_1"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_1"),
            conv_block(128, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="down_block_2"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_2"),
            conv_block(256, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="down_block_3"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_3"),
            conv_block(512, 3, 1, dropout_rate=dropout_rate, name="down_block_4"),
            tf.keras.Sequential([tf.keras.layers.MaxPool2D((2, 2), (2, 2))], name="max_pool_4"),
        ]

        self.bottleneck = conv_block(1024, 3, 1, dropout_rate=dropout_rate, name="bottleneck")

        self.expansive_path = [
            upsample(filters=512, size=2, dropout_rate=after_conv_block_dropout_rate,
                     upsampling_method=upsampling_method,
                     name="up_conv_1"),
            conv_block(512, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="conv_block_right_1"),
            upsample(256, 2, dropout_rate=after_conv_block_dropout_rate, upsampling_method=upsampling_method,
                     name="up_conv_2"),
            conv_block(256, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="conv_block_right_2"),
            upsample(128, 2, dropout_rate=after_conv_block_dropout_rate, upsampling_method=upsampling_method,
                     name="up_conv_3"),
            conv_block(128, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="conv_block_right_3"),
            upsample(64, 2, dropout_rate=after_conv_block_dropout_rate, upsampling_method=upsampling_method,
                     name="up_conv_4"),
            conv_block(64, 3, 1, dropout_rate=after_conv_block_dropout_rate, name="conv_block_right_4"),
        ]

        self.conv_out = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same', activation=None)
        self.crop_output = tf.keras.layers.Cropping2D(((2, 2), (2, 2)))

    def call(self, inputs, training=None, mask=None):
        x = tf.pad(inputs, ((0, 0), (94, 94), (94, 94), (0, 0)), mode="SYMMETRIC")

        # Downsampling and saving outputs to establish skip connection
        skips = []
        for block in self.contracting_path:
            x = block(x)
            if "max_pool" not in block.name and len(skips) < 4:
                skips.append(x)
        skips = list(reversed(skips))

        x = self.bottleneck(x)

        # Upsampling, cropping and concatenation with skip connections
        counter = 0
        for block in self.expansive_path:
            x = block(x)
            if "up_conv" in block.name and counter < 4:
                concat = tf.keras.layers.Concatenate()
                x = concat([crop_to_fit(x, skips[counter]), x])
                counter += 1

        # convert to output
        logits = self.conv_out(x)
        cropped_logits = self.crop_output(logits)

        return cropped_logits


def crop_to_fit(target_tensor, to_crop):
    """
    Crops an image to width and height of target tensor.
    Args:
        target_tensor: tensor with desired shape [batch, new_height, new_width, channels]
        to_crop: tensor which should be cropped to shape of target_tensor, shape is [batch, old_height, old_width, channels]

    Returns:
        Cropped tensor of shape [batch, new_height, new_width, channels]

    """
    return tf.image.resize_with_crop_or_pad(to_crop, tf.shape(target_tensor)[1], tf.shape(target_tensor)[2])


def conv_block(filters: int,
               size,
               stride=(1, 1),
               use_batch_norm: bool = True,
               dropout_rate=None,
               name: str = None) -> tf.keras.Sequential:
    """
    Conv2D => ReLu => Conv2D => ReLu => (Dropout)

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
                                          kernel_initializer='he_normal', use_bias=True))

        if use_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.ReLU())
    if dropout_rate:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    return result


def upsample(filters,
             size,
             stride=(2, 2),
             use_batch_norm: bool = True,
             name: str = None,
             dropout_rate=None,
             upsampling_method: str = 'transpose') -> tf.keras.Sequential:
    """
    Conv2DTranspose => (BatchNorm) => ReLu

    :param filters: number of filters
    :param size: size of filters
    :param stride: stride for conv2D
    :param use_batch_norm: if True batch normalization is applied after each conv layer
    :param name: optional a name for the upsampling block
    :param dropout_rate: either None (no dropout) or float between 0 and 1 (apply dropout)
    :param upsampling_method: either "upsampling" = Upsampling2D or "transpose" = Conv2DTranspose in the expansive path
    :return: up sample block as sequential model
    """
    result = tf.keras.Sequential(name=name)
    if upsampling_method == 'upsampling':
        result.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        result.add(
            tf.keras.layers.Conv2D(filters, 2, activation='relu', padding='same', kernel_initializer='he_normal'))
    elif upsampling_method == 'transpose':
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size,
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False))
    else:
        raise ValueError("Unknown upsampling_method: {}".format(upsampling_method))
    if use_batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    if dropout_rate:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    return result
