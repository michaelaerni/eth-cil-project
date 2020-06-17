import tensorflow as tf
import typing

"""
Model parts and losses for MoCo and the improved MoCo framework based on
Momentum Contrast for Unsupervised Visual Representation Learning (arXiv:1911.05722 [cs.CV])
and
Improved Baselines with Momentum Contrastive Learning (arXiv:2003.04297 [cs.CV])
"""


class EncoderMoCoTrainingModel(tf.keras.Model):
    """
    Helper model which handles MoCo training of a given encoder model.
    """

    def __init__(
            self,
            encoder: tf.keras.Model,
            momentum_encoder: tf.keras.Model,
            momentum: float
    ):
        """
        Create a new MoCo encoder training model.

        Args:
            encoder: Encoder to train using momentum contrastive learning.
            momentum_encoder: Different encoder to be updated using momentum.
                This is required since TensorFlow supports cloning subclassed models only starting version 2.2.
            momentum: Momentum for encoder updates each batch in [0, 1)
        """

        super(EncoderMoCoTrainingModel, self).__init__()

        if not (0 <= momentum < 1):
            raise ValueError(f'Momentum must be in [0, 1) but is {momentum}')

        self.encoder = encoder
        self.momentum_encoder = momentum_encoder

        # TODO: Idelly we would like to set the momentum encoder to be not trainable.
        #  However, that breaks updating its weights as seen e.g. in the open issue
        #  https://github.com/keras-team/keras/issues/6607

        self.momentum = momentum

    def build(self, input_shape):
        super(EncoderMoCoTrainingModel, self).build(input_shape)

        # Initialize momentum encoder weights to weights of current model
        self.update_momentum_encoder()

    def call(self, inputs, training=None, mask=None):
        """
        Call this model.

        Args:
            inputs: Tuple of two tensors, each representing the same batch of images but augmented differently.
            training: Additional argument, unused.
            mask: Additional argument, unused.

        Returns:
            TODO
        """
        query_inputs, key_inputs = tf.unstack(inputs)
        # TODO: Emulate shuffling BN in some form
        # TODO: Everything here
        a = self.encoder(query_inputs)
        b = self.momentum_encoder(key_inputs)
        return a + b

    def update_momentum_encoder(self):
        """
        Update the weights of the momentum encoder with the weights from the trainable encoder using momentum.
        """
        # TODO: Check whether this is fast or slow
        new_weights = self.encoder.get_weights()
        old_weights = self.momentum_encoder.get_weights()

        updated_weights = list(
            self.momentum * old_weight + (1.0 - self.momentum) * new_weight
            for (old_weight, new_weight) in zip(old_weights, new_weights)
        )
        self.momentum_encoder.set_weights(updated_weights)


class FCHead(tf.keras.layers.Layer):
    """
    TODO: All documentation
    """

    def __init__(
            self,
            backbone: tf.keras.Model,
            features: int,
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',  # TODO: What initializer to use?
            weight_decay: float = 1e-4,
            **kwargs
    ):
        super(FCHead, self).__init__(**kwargs)

        self.backbone = backbone

        # Global average pooling
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        # Output fully connected layer creating the features
        self.fc = tf.keras.layers.Dense(
            features,
            activation=None,
            kernel_initializer=dense_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay)
        )

    def call(self, inputs, **kwargs):
        # Generate output features from backbone
        input_features = self.backbone(inputs)
        pooled_features = self.pool(input_features[-1])
        unscaled_output_features = self.fc(pooled_features)

        # Normalize features to have unit norm
        output_features = unscaled_output_features / tf.norm(
            unscaled_output_features,
            ord='euclidean',
            axis=-1,
            keepdims=True
        )
        return output_features
