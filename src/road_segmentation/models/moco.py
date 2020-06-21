import typing

import tensorflow as tf

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

        # TODO: [v1] It is not quite clear how batch normalization on the momentum encoder works
        #  and there are three possibilities (frozen, inference, training).
        #  We need to investigate which one is happening.
        self._momentum_update_callback = None

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

        # First, match all weights via their order.
        # This is the only reliable way to match weights since the names are always different.
        # That requires temporarily setting the encoder to non-trainable to have the weight order match up.
        self.encoder.trainable = False
        encoder_weights_by_momentum_name_map = {
            momentum_weight.name: encoder_weight
            for (momentum_weight, encoder_weight) in zip(self.momentum_encoder.weights, self.encoder.weights)
        }
        self.encoder.trainable = True

        # Then, match the weights remaining in the momentum encoder to the normal encoder
        weight_mapping = [
            (momentum_weight, encoder_weights_by_momentum_name_map[momentum_weight.name])
            for momentum_weight in self.momentum_encoder.weights
        ]
        # TODO: [v1] Make sure the mapping is actually correct!

        # Finally, update the callback which performs the momentum updates
        self._momentum_update_callback = self._UpdateMomentumEncoderCallback(weight_mapping, self.momentum)

    def call(self, inputs, training=None, mask=None):
        """
        Call this model.

        Args:
            inputs: Tuple of two tensors, each representing the same batch of images but augmented differently.
            training: Additional argument, unused.
            mask: Additional argument, unused.

        Returns:
            Batch of logit predictions which keys belong to the query. The true key logits are always at index 0.
        """
        query_inputs, key_inputs = tf.unstack(inputs)

        # Calculate features for queries and positive keys
        # TODO: [v1] Emulate shuffling BN in some form
        #  or replace with layer norm as in https://arxiv.org/abs/1905.09272
        query_features = self.encoder(query_inputs)
        key_features_positive = self.momentum_encoder(key_inputs)

        # Prevent gradient back to the keys
        key_features_positive = tf.keras.backend.stop_gradient(key_features_positive)

        # TODO: Allow similarity measures other than the dot product?

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
        logits = (1.0 / self.temperature) * logits

        # Update queue values and pointer
        # Note that both updates implicitly assume the queue size to be a multiple of the batch size
        batch_size = tf.shape(key_features_positive)[0]
        queue_size = tf.shape(self.queue)[0]
        with tf.control_dependencies([key_features_positive]):
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
                    # TODO: [v1] Make sure the gradient calculation uses the old queue value, not the new one!
                    logits = tf.identity(logits)

        return logits

    def create_callbacks(self) -> typing.List[tf.keras.callbacks.Callback]:
        """
        Creates Keras callbacks which are required to train this model correctly.

        Returns:
            List of callbacks which should be appended to the list of Keras training callbacks.
        """

        return [
            self._momentum_update_callback
        ]

    class _UpdateMomentumEncoderCallback(tf.keras.callbacks.Callback):
        def __init__(
                self,
                weight_mapping: typing.List[typing.Tuple[tf.Variable, tf.Variable]],
                momentum: float
        ):
            super().__init__()

            self.weight_mapping = weight_mapping
            self.momentum = momentum

        def on_train_begin(self, logs=None):
            # Initially set momentum encoder weights to be equal to the encoder weights
            for momentum_weight, encoder_weight in self.weight_mapping:
                momentum_weight.assign(encoder_weight)

        def on_train_batch_end(self, batch, logs=None):
            # Assign weights of encoder with momentum
            for momentum_weight, encoder_weight in self.weight_mapping:
                # This formulation is equivalent but slightly faster
                # m * me + (1 - m) * e = me + (m - 1) * me + (1 - m) * e = me + (1 - m) * (e - me)
                momentum_weight.assign_add((encoder_weight - momentum_weight) * (1.0 - self.momentum))


class FCHead(tf.keras.layers.Layer):
    """
    MoCo v1 head which transforms the outputs of a backbone model
    into global representations to be used in contrastive learning.
    """

    def __init__(
            self,
            backbone: tf.keras.Model,
            features: int,
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'glorot_uniform',  # TODO: What initializer to use?
            weight_decay: float = 1e-4,
            **kwargs
    ):
        """
        Create a new MoCo v1 head.

        Args:
            backbone: Backbone model to generate the representations from.
                Should return a list of tensors. The representation is generated from the last one.
            features: Dimensionality of the resulting representation.
            dense_initializer: Weight initializer for dense layers.
            weight_decay: Weight decay for dense layer weights.
            **kwargs: Additional arguments passed to tf.keras.layers.Layer.
        """

        super(FCHead, self).__init__(**kwargs)

        self.backbone = backbone
        self.backbone.trainable = self.trainable

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
