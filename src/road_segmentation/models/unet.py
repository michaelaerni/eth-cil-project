import numpy as np
import tensorflow as tf
import typing


class UNet(tf.keras.Model):
    """
    Implementation of plain U-Net according to original paper (https://arxiv.org/abs/1505.04597).
    """

    def __init__(self,
                 dropout_rate: float,
                 apply_dropout: bool,
                 upsampling_method: str,
                 number_of_filters: int,
                 number_of_scaling_steps: int,
                 apply_batch_norm: bool,
                 input_padding: typing.Tuple[
                     typing.Tuple[int, int],
                     typing.Tuple[int, int],
                     typing.Tuple[int, int],
                     typing.Tuple[int, int]],
                 output_cropping: typing.Tuple[
                     typing.Tuple[int, int],
                     typing.Tuple[int, int]]):
        super(UNet, self).__init__()

        after_conv_block_dropout_rate = None
        if apply_dropout:
            after_conv_block_dropout_rate = dropout_rate

        self.number_of_scaling_steps = number_of_scaling_steps

        self.input_padding = lambda x: tf.pad(x, input_padding, mode="REFLECT")

        self.contracting_path = []
        for i in range(self.number_of_scaling_steps):
            current_number_of_filters = np.power(2, i) * number_of_filters
            print(current_number_of_filters)
            conv_block_ = conv_block(filters=current_number_of_filters,
                                     size=(3, 3),
                                     stride=(1, 1),
                                     apply_batch_norm=apply_batch_norm,
                                     dropout_rate=after_conv_block_dropout_rate,
                                     name=f"down_block_{i + 1}")
            self.contracting_path.append(conv_block_)
            self.contracting_path.append(tf.keras.layers.MaxPool2D((2, 2), (2, 2), name=f"max_pool_{i + 1}"))

        current_number_of_filters = np.power(2, self.number_of_scaling_steps) * number_of_filters

        self.bottleneck = conv_block(current_number_of_filters,
                                     size=(3, 3),
                                     stride=(1, 1),
                                     apply_batch_norm=apply_batch_norm,
                                     dropout_rate=dropout_rate,
                                     name="bottleneck")

        self.expansive_path = []

        for i in range(1, self.number_of_scaling_steps + 1):
            current_number_of_filters = np.power(2, self.number_of_scaling_steps - i) * number_of_filters
            print(current_number_of_filters)
            upsampling_block = upsample(filters=current_number_of_filters,
                                        size=(2, 2),
                                        stride=(2, 2),
                                        apply_batch_norm=apply_batch_norm,
                                        dropout_rate=after_conv_block_dropout_rate,
                                        upsampling_method=upsampling_method,
                                        name=f"up_conv_{i}")
            conv_block_ = conv_block(filters=current_number_of_filters,
                                     size=(3, 3),
                                     stride=(1, 1),
                                     apply_batch_norm=apply_batch_norm,
                                     dropout_rate=after_conv_block_dropout_rate,
                                     name=f"conv_block_right_{i}")
            self.expansive_path.append(upsampling_block)
            self.expansive_path.append(conv_block_)

        self.conv_out = tf.keras.layers.Conv2D(filters=1,
                                               kernel_size=(1, 1),
                                               strides=(1, 1),
                                               padding='same',
                                               activation=None)
        self.crop_output = tf.keras.layers.Cropping2D(output_cropping)

    def call(self, inputs, training=None, mask=None):
        x = self.input_padding(inputs)

        # Downsampling and saving outputs to establish skip connection
        skips = []
        for i, block in enumerate(self.contracting_path):
            x = block(x)
            if i % 2 == 0 and len(skips) < self.number_of_scaling_steps:
                # if "max_pool" not in block.name and len(skips) < self.number_of_scaling_steps:
                skips.append(x)

            assert (i % 2 == 0 and len(skips) < self.number_of_scaling_steps) == (
                    "max_pool" not in block.name and len(skips) < self.number_of_scaling_steps)
        skips = list(reversed(skips))

        x = self.bottleneck(x)

        # Upsampling, cropping and concatenation with skip connections
        counter = 0
        for i, block in enumerate(self.expansive_path):
            x = block(x)

            assert ("up_conv" in block.name and counter < self.number_of_scaling_steps) == (
                    i % 2 == 0 and counter < self.number_of_scaling_steps)
            if i % 2 == 0 and counter < self.number_of_scaling_steps:
                # if "up_conv" in block.name and counter < self.number_of_scaling_steps:
                x = tf.concat([crop_to_fit(x, skips[counter]), x], axis=-1)
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
        to_crop: tensor which should be cropped to shape of target_tensor,
                 shape is [batch, old_height, old_width, channels]

    Returns:
        Cropped tensor of shape [batch, new_height, new_width, channels]

    """
    return tf.image.resize_with_crop_or_pad(to_crop, tf.shape(target_tensor)[1], tf.shape(target_tensor)[2])


def conv_block(filters: int,
               size: typing.Tuple[int, int],
               stride: typing.Tuple[int, int] = (1, 1),
               apply_batch_norm: bool = False,
               dropout_rate: float = None,
               name: str = None) -> tf.keras.Model:
    """
    Conv2D => (BN) => ReLu => Conv2D => (BN) => ReLu => (Dropout)

    Args:
        filters: number of filters
        size: size of ilters
        stride: stride for Conv2D
        apply_batch_norm: if True batch normalization is applied after each conv layer
        dropout_rate: either None (no dropout) or float between 0 and 1 (apply dropout)
        name: optional a name for the conv block

    Returns:
        convolution block as keras model
    """
    result = tf.keras.Sequential(name=name)
    for i in range(2):
        result.add(tf.keras.layers.Conv2D(filters,
                                          size,
                                          strides=stride,
                                          padding='valid',
                                          kernel_initializer='he_normal',
                                          use_bias=True))

        if apply_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.ReLU())
    if dropout_rate:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    return result


def upsample(filters: int,
             size: typing.Tuple[int, int],
             stride: typing.Tuple[int, int] = (2, 2),
             apply_batch_norm: bool = False,
             name: str = None,
             dropout_rate: float = None,
             upsampling_method: str = 'transpose') -> tf.keras.Model:
    """
    Conv2DTranspose => (BatchNorm) => ReLu => (Dropout)
    or
    UpSampling2D => Conv2D => (BatchNorm) => ReLu => (Dropout)

    Args:
        filters: number of filters
        size: size of filters
        stride: stride for Conv2D
        apply_batch_norm: if True batch normalization is applied after each conv layer
        name: optional a name for the upsampling block
        dropout_rate: either None (no dropout) or float between 0 and 1 (apply dropout)
        upsampling_method: "upsampling" for upsampling via interpolation or "transpose" for learnable upsampling

    Returns:
        up sample block
    """
    result = tf.keras.Sequential(name=name)
    if upsampling_method == 'upsampling':
        result.add(tf.keras.layers.UpSampling2D(size=(2, 2)))
        result.add(
            tf.keras.layers.Conv2D(filters,
                                   kernel_size=(2, 2),
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal'))
    elif upsampling_method == 'transpose':
        result.add(
            tf.keras.layers.Conv2DTranspose(filters,
                                            size,
                                            strides=stride,
                                            padding='same',
                                            kernel_initializer='he_normal',
                                            use_bias=False))
    else:
        raise ValueError("Unknown upsampling_method: {}".format(upsampling_method))
    if apply_batch_norm:
        result.add(tf.keras.layers.BatchNormalization())
    if dropout_rate:
        result.add(tf.keras.layers.Dropout(dropout_rate))

    return result
