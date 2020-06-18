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
            momentum: float,
            temperature: float,
            queue_size: int,
            features: int
    ):
        """
        Create a new MoCo encoder training model.

        Args:
            encoder: Encoder to train using momentum contrastive learning.
            momentum_encoder: Different encoder to be updated using momentum.
                This is required since TensorFlow supports cloning subclassed models only starting version 2.2.
            momentum: Momentum for encoder updates each batch in [0, 1)
            temperature: Temperature for outputs.
            queue_size: Size of the queue containing previous key features.
            features: Dimensionality of the representations.
        """

        super(EncoderMoCoTrainingModel, self).__init__()

        if not (0 <= momentum < 1):
            raise ValueError(f'Momentum must be in [0, 1) but is {momentum}')

        self.encoder = encoder
        self.momentum_encoder = momentum_encoder

        # TODO: Ideally we would like to set the momentum encoder to be not trainable.
        #  However, that breaks updating its weights as seen e.g. in the open issue
        #  https://github.com/keras-team/keras/issues/6607
        #  It is not quite clear how batch normalization on the momentum encoder works
        #  and there are three possibilities (frozen, inference, training).
        #  We need to investigate which one is happening.

        self.momentum = momentum

        self.temperature = temperature

        queue_shape = (queue_size, features)
        self.queue: tf.Variable = self.add_weight(
            name='queue',
            shape=queue_shape,
            dtype=tf.float32,
            initializer=lambda shape, dtype: tf.math.l2_normalize(tf.random.normal(shape, dtype=dtype)),  # Initialize queue with random features
            trainable=False
        )
        self.queue_pointer: tf.Variable = self.add_weight(
            name='queue_pointer',
            shape=[],
            dtype=tf.int32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )

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

        # Calculate features for queries and positive keys
        # TODO: Emulate shuffling BN in some form
        query_features = self.encoder(query_inputs)
        key_features_positive = self.momentum_encoder(key_inputs)

        # Prevent gradient back to the keys
        key_features_positive = tf.keras.backend.stop_gradient(key_features_positive)

        # TODO: Allow similarity measures other than the dot product

        # Positive logits
        logits_positive = tf.matmul(
            tf.expand_dims(query_features, axis=1),  # => (batch size, 1, MoCo dim)
            tf.expand_dims(key_features_positive, axis=-1)  # => (batch size, MoCo dim, 1)
        )
        logits_positive = tf.squeeze(logits_positive, axis=-1)  # (batch size, 1, 1) => (batch size, 1)

        # Negative logits
        logits_negative = tf.matmul(
            query_features,  # => (batch size, MoCo dim)
            self.queue,  # => (queue size, MoCo dim)^T
            transpose_b=True
        )

        # Combine logits such that index 0 is the positive instance
        logits = tf.concat((logits_positive, logits_negative), axis=-1)  # => (batch size, queue size + 1)

        # Apply temperature
        logits = logits / self.temperature

        # Update queue values and pointer
        batch_size = tf.shape(key_features_positive)[0]
        queue_size = tf.shape(self.queue)[0]
        # TODO: Both updates implicitly assume the queue size to be a multiple of the batch size
        with tf.control_dependencies([
            self.queue[self.queue_pointer:self.queue_pointer + batch_size, :].assign(key_features_positive)
        ]):
            # Only update queue pointer *after* updating the queue itself
            with tf.control_dependencies([
                self.queue_pointer.assign(tf.math.mod(self.queue_pointer + batch_size, queue_size))
            ]):
                # Dummy op to ensure updates are applied
                # The operations in the outer tf.control_dependencies scopes are performed *before* the identity op.
                # Since logits are returned and further used this ensures that the queue is always updated.
                # TODO: Make sure the gradient calculation uses the old queue value, not the new one!
                logits = tf.identity(logits)

        return logits

    def update_momentum_encoder(self):
        """
        Update the weights of the momentum encoder with the weights from the trainable encoder using momentum.
        """
        # TODO: Check whether this is fast or slow
        new_weights = self.encoder.get_weights()
        old_weights = self.momentum_encoder.get_weights()

        # TODO: Try setting trainable = True and trainable = False on momentum_encoder but check with BN layers

        updated_weights = list(
            self.momentum * old_weight + (1.0 - self.momentum) * new_weight
            for (old_weight, new_weight) in zip(old_weights, new_weights)
        )
        self.momentum_encoder.set_weights(updated_weights)

    def create_callbacks(self) -> typing.List[tf.keras.callbacks.Callback]:
        return [
            # Update momentum encoder after each training batch
            tf.keras.callbacks.LambdaCallback(
                on_batch_end=lambda batch, logs: self.update_momentum_encoder()
            )
        ]


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
        output_features = tf.math.l2_normalize(unscaled_output_features, axis=-1)
        return output_features
