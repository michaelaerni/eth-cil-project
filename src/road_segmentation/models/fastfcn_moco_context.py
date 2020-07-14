import typing

import tensorflow as tf


class FastFCNMocoContextTrainingModel(tf.keras.Model):
    """
    Helper model which handles training a FastFCN with context encoding module, using
    MoCo loss instead of semantic encoding loss.
    """

    def __init__(
            self,
            encoder: tf.keras.Model,
            momentum_encoder: tf.keras.Model,
            momentum: float,
            temperature: float,
            queue_size: int,
            semantic_features: int
    ):
        """
        Create a new FastFCN MoCo context training model.

        Args:
            encoder: Encoder to train using momentum contrastive learning.
            momentum_encoder: Different encoder to be updated using momentum.
                This is required since TensorFlow supports cloning subclassed models only starting version 2.2.
            momentum: Momentum for encoder updates each batch in [0, 1)
            temperature: Temperature for outputs.
            queue_size: Size of the queue containing previous key features.
            semantic_features: Number of semantic features provided by the context encoding module.
        """

        super(FastFCNMocoContextTrainingModel, self).__init__()

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

        queue_shape = (queue_size, semantic_features)
        self.queue: tf.Variable = self.add_weight(
            name='queue',
            shape=queue_shape,
            dtype=tf.float32,
            initializer=lambda shape, dtype: tf.math.l2_normalize(tf.random.normal(shape, dtype=dtype)),
            # Initialize queue with random features
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
        super(FastFCNMocoContextTrainingModel, self).build(input_shape)

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

        # Finally, update the callback which performs the momentum updates
        self._momentum_update_callback = self._UpdateMomentumEncoderCallback(weight_mapping, self.momentum)

    def call(self, inputs, training=False, mask=None):
        """
        Call this model.

        Args:
            inputs: Tuple of three tensors. First is a labelled batch used for learning segmentation masks,
             the last two are from the same unlabelled input but augmented differently, used for learning internal
             representations using contrastive loss.
            training: Additional argument, unused.
            mask: Additional argument, unused.

        Returns:
            Batch of logit predictions which keys belong to the query. The true key logits are always at index 0.
        """
        labelled_inputs, query_inputs, key_inputs = inputs

        # Predict the segmentation mask
        masks, _ = self.encoder(labelled_inputs)

        # Calculate features for queries and positive keys
        _, query_features = self.encoder(query_inputs)
        _, key_features_positive = self.momentum_encoder(key_inputs)

        # Prevent gradient back to the keys
        key_features_positive = tf.keras.backend.stop_gradient(key_features_positive)

        # Normalize representations
        query_features = tf.math.l2_normalize(query_features, axis=-1)
        key_features_positive = tf.math.l2_normalize(key_features_positive, axis=-1)

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

        return masks, logits

    def create_callbacks(self) -> typing.List[tf.keras.callbacks.Callback]:
        """
        Creates Keras callbacks which are required to train this model correctly.

        Returns:
            List of callbacks which should be appended to the list of Keras training callbacks.
        """

        return [
            self._momentum_update_callback
        ]

    def get_prediction_model(self):
        """
        Return the model which only returns segmentation predictions and semantic encodings.
        Returns:
            Model for prediction.
        """
        return self.encoder

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
