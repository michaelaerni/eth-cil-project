import abc
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


class SpatialEncoderMoCoTrainingModel(tf.keras.Model):
    """
    Helper model which handles MoCo training of a given encoder model on a spatial pretext task.
    """

    def __init__(
            self,
            encoder: 'Base2DHead',
            momentum_encoder: 'Base2DHead',
            momentum: float,
            temperature: float,
            queue_size: int,
            features: int
    ):
        """
        Create a new spatial MoCo encoder training model.

        Args:
            encoder: Encoder to train using momentum contrastive learning.
            momentum_encoder: Different encoder to be updated using momentum.
                This is required since TensorFlow supports cloning subclassed models only starting version 2.2.
            momentum: Momentum for encoder updates each batch in [0, 1)
            temperature: Temperature for outputs.
            queue_size: Size of the queue containing previous key features.
            features: Dimensionality of the representations.
        """

        # TODO: Maybe we can merge this with the other MoCo trainer

        super(SpatialEncoderMoCoTrainingModel, self).__init__()

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

        self.features = features

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
        super(SpatialEncoderMoCoTrainingModel, self).build(input_shape)

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

    def call(self, inputs, training=None, mask=None):
        """
        Call this model.

        Args:
            inputs: Tuple of three tensors.
             The first two represent a batch of query and key images respectively.
             The third one represents a batch of augmentation parameters for later undoing.
            training: Additional argument, unused.
            mask: Additional argument, unused.

        Returns:
            Batch of logit predictions which keys belong to the query. The true key logits are always at index 0.
        """
        # Unpack images and augmentation information
        query_inputs = inputs[0]
        key_inputs = inputs[1]
        query_offset_x, query_offset_y, key_offset_x, key_offset_y, key_flipped, key_rotations = inputs[2:]

        # Calculate features for queries and positive keys
        query_features = self.encoder(
            (query_inputs, query_offset_x, query_offset_y, key_offset_x, key_offset_y)
        )
        key_features_positive = self.momentum_encoder(
            (key_inputs, key_offset_x, key_offset_y, query_offset_x, query_offset_y, key_flipped, key_rotations)
        )

        # Prevent gradient back to the keys
        key_features_positive = tf.keras.backend.stop_gradient(key_features_positive)

        # TODO: Allow similarity measures other than the dot product

        # Positive logits
        logits_positive = tf.matmul(
            tf.expand_dims(query_features, axis=2),  # => (batch size, locations, 1, MoCo dim)
            tf.expand_dims(key_features_positive, axis=-1)  # => (batch size, locations, MoCo dim, 1)
        )
        # (batch size, locations, 1, 1) => (batch size, locations, 1)
        logits_positive = tf.squeeze(logits_positive, axis=-1)

        # Negative logits
        logits_negative = tf.matmul(
            query_features,  # => (batch size, locations, MoCo dim)
            self.queue,  # => (queue size, MoCo dim)^T
            transpose_b=True
        )  # => (batch size, locations, queue size)

        # Combine logits over locations such that index 0 is the positive instance
        logits = tf.concat((logits_positive, logits_negative), axis=-1)  # => (batch size, locations, queue size + 1)

        # TODO: There are multiple possible ways to use the logits here.
        #  One would be to flatten all locations into (batch size, locations * (queue size + 1)) and
        #  use the sigmoid cross entropy loss, enforcing all matching locations to be similar.
        #  This treats each element in the batch as a single sample.
        #  Another would be to combine all locations in a batch into a "super batch"
        #  (batch size * locations, queue size + 1) and using the softmax cross entropy loss.
        #  This treats each location as an individual sample.
        #  Here, the latter version is used since it is more easily interpretable w.r.t. the InfoNCE loss.
        queue_size = tf.shape(self.queue)[0]
        logits = tf.reshape(logits, (-1, queue_size + 1))

        # Apply temperature
        logits = (1.0 / self.temperature) * logits

        # Update queue values and pointer
        # Note that both updates implicitly assume the queue size to be a multiple of the batch size * num locations
        enqueuing_features = tf.reshape(key_features_positive, (-1, self.features))
        num_enqueuing_features = tf.shape(enqueuing_features)[0]
        with tf.control_dependencies([enqueuing_features]):
            with tf.control_dependencies([
                self.queue[self.queue_pointer:self.queue_pointer + num_enqueuing_features, :].assign(
                    enqueuing_features
                )
            ]):
                # Only update queue pointer *after* updating the queue itself
                with tf.control_dependencies([
                    self.queue_pointer.assign(tf.math.mod(self.queue_pointer + num_enqueuing_features, queue_size))
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
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Create a new MoCo v1 head.

        Args:
            backbone: Backbone model to generate the representations from.
                Should return a list of tensors. The representation is generated from the last one.
            features: Dimensionality of the resulting representation.
            dense_initializer: Weight initializer for dense layers.
            kernel_regularizer: Regularizer for dense layer weights.
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
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, **kwargs):
        # Generate output features from backbone
        input_features = self.backbone(inputs)
        pooled_features = self.pool(input_features)
        unscaled_output_features = self.fc(pooled_features)

        # Normalize features to have unit norm
        output_features = tf.math.l2_normalize(unscaled_output_features, axis=-1)
        return output_features


class MLPHead(tf.keras.layers.Layer):
    """
    MoCo v2 head which transforms the outputs of a backbone model
    into global representations to be used in contrastive learning.
    This head contains one additional hidden layer compared to the original v1 head.
    """

    def __init__(
            self,
            backbone: tf.keras.Model,
            output_features: int,
            intermediate_features: int,
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        """
        Create a new MoCo v2 head.

        Args:
            backbone: Backbone model to generate the representations from.
                Should return a list of tensors. The representation is generated from the last one.
            output_features: Dimensionality of the resulting representation.
            intermediate_features: Dimensionality of the preliminary layer's output. This should be the same as input.
            dense_initializer: Weight initializer for dense layers.
            kernel_regularizer: Regularizer for dense layer weights.
            **kwargs: Additional arguments passed to tf.keras.layers.Layer.
        """

        super(MLPHead, self).__init__(**kwargs)

        self.backbone = backbone
        self.backbone.trainable = self.trainable

        # Global average pooling
        self.pool = tf.keras.layers.GlobalAveragePooling2D()

        # Intermediate hidden layer which (usually) keeps the backbone's output dimensionality
        self.intermediate = tf.keras.layers.Dense(
            intermediate_features,
            activation='relu',  # No normalization, thus ReLU can be performed as part of the layer
            use_bias=True,
            kernel_initializer=dense_initializer,
            kernel_regularizer=kernel_regularizer
        )

        # Output fully connected layer creating the features
        self.fc = tf.keras.layers.Dense(
            output_features,
            activation=None,
            kernel_initializer=dense_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call(self, inputs, **kwargs):
        # Generate intermediate features from backbone
        input_features = self.backbone(inputs)
        pooled_features = self.pool(input_features[-1])
        intermediate_features = self.intermediate(pooled_features)

        # Generate output features and normalize to unit norm
        unscaled_output_features = self.fc(intermediate_features)
        output_features = tf.math.l2_normalize(unscaled_output_features, axis=-1)
        return output_features


class Base2DHead(tf.keras.layers.Layer, metaclass=abc.ABCMeta):
    """
    Base class for spatial MoCo heads.
    """

    def __init__(
            self,
            backbone: tf.keras.Model,
            feature_rectangle_size: int,
            undo_spatial_transformations: bool,
            **kwargs
    ):
        # TODO: Document
        super(Base2DHead, self).__init__(**kwargs)

        self.backbone = backbone
        self.backbone.trainable = self.trainable

        self.feature_rectangle_size = feature_rectangle_size
        self.undo_spatial_transformations = undo_spatial_transformations

    def call(self, inputs, **kwargs):
        # Unpack images and augmentation information
        image = inputs[0]
        offset_x, offset_y, other_offset_x, other_offset_y = tf.unstack(tf.cast(inputs[1:5], tf.int32))

        # Calculate full sized features using backbone
        features_full = self.backbone(image)[-1]

        # Undo rotation and flip on key features
        if self.undo_spatial_transformations:
            was_flipped = tf.cast(inputs[5], tf.bool)
            num_rotations = tf.cast(inputs[6], tf.int32)
            features_full = self._undo_transformations(features_full, was_flipped, num_rotations)

        # Align features with other (separately generated) feature map and crop accordingly
        # FIXME: Currently, offsets are calculated twice. However, those are very small and fast calculations
        #  and thus should not matter too much.
        intermediate_features = self._align_features(
            features_full,
            offset_x,
            offset_y,
            other_offset_x,
            other_offset_y
        )

        # Apply output transformations
        return self.call_output(intermediate_features, **kwargs)

    @abc.abstractmethod
    def call_output(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Call method of child classes implementing concrete heads.
        This should transform the selected and aligned features from the backbone into a representation
        which is compared using some contrastive loss.

        Args:
            inputs: 3 tensor of encoded and aligned features with shape (batch size, num locations, backbone features)
            **kwargs: Additional arguments

        Returns:
            Representations as a 3 tensor of shape (batch size, num locations, representation features)
        """
        pass

    @classmethod
    def _undo_transformations(
            cls,
            features: tf.Tensor,
            was_flipped_values: tf.Tensor,
            rotations_values: tf.Tensor
    ) -> tf.Tensor:
        # Need to map the transformations since tf.image.* usually only supports operations on a single image
        def _undo_single(inputs: typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
            current_features, was_flipped, rotations = inputs

            current_features = tf.image.rot90(
                current_features,
                tf.constant(4) - rotations  # Remaining rotations to have a full 360 degree rotation
            )
            current_features = tf.cond(
                was_flipped,
                lambda: tf.image.flip_left_right(current_features),
                lambda: current_features
            )

            return current_features

        return tf.map_fn(
            _undo_single,
            (features, tf.squeeze(was_flipped_values), tf.squeeze(rotations_values)),
            dtype=tf.float32
        )

    def _align_features(
            self,
            features_full: tf.Tensor,
            offset_x: tf.Tensor,
            offset_y: tf.Tensor,
            other_offset_x: tf.Tensor,
            other_offset_y: tf.Tensor
    ) -> tf.Tensor:
        # FIXME: Make sure the calculations are correct!!

        # Squeeze inputs to ensure they are rank 1
        offset_x = tf.squeeze(offset_x)
        offset_y = tf.squeeze(offset_y)
        other_offset_x = tf.squeeze(other_offset_x)
        other_offset_y = tf.squeeze(other_offset_y)

        # TODO: Experiment with different overlaps (e.g. taking random features, taking all, etc)
        #  Currently takes a random region of fixed size from the overlapping rectangle

        # All calculations are in the stride space of the original full input image unless mentioned!
        # They are in the end transformed relative to the patch received by this encoder.

        # Determine overlapping rectangle via offsets
        min_offset_x = tf.minimum(offset_x, other_offset_x)
        max_offset_x = tf.maximum(offset_x, other_offset_x)
        min_offset_y = tf.minimum(offset_y, other_offset_y)
        max_offset_y = tf.maximum(offset_y, other_offset_y)

        # This implicitly assumes both query and key feature shapes to be equal!
        batch_size, feature_height, feature_width, num_features = tf.unstack(tf.shape(features_full))

        # Determine sampling region
        # This implicitly assumes that there always exists a feasible region
        max_x = min_offset_x + feature_width - self.feature_rectangle_size
        max_y = min_offset_y + feature_height - self.feature_rectangle_size

        # Sample and return random rectangle of features (in stride space of input image)
        global_crop_x = self._sample_from_ranges(max_offset_x, max_x + 1)
        global_crop_y = self._sample_from_ranges(max_offset_y, max_y + 1)

        # Convert the coordinates into the respective patch's stride spaces and crop
        crop_x = global_crop_x - offset_x
        crop_y = global_crop_y - offset_y
        features_cropped = self._crop_feature_squares(features_full, crop_x, crop_y)

        # Reshape correctly into a 3 tensor
        output_shape = (batch_size, self.feature_rectangle_size * self.feature_rectangle_size, num_features)
        output = tf.reshape(features_cropped, output_shape)

        return output

    @classmethod
    def _sample_from_ranges(cls, minimums: tf.Tensor, upper_bounds: tf.Tensor) -> tf.Tensor:
        # Minimum and maximum are rank 1 tensors
        # TODO: This is a bit of a hack since tf.random.uniform cannot handle rank 1 bounds.
        def _sample_single(inputs: typing.Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
            minval, maxval = inputs
            return tf.random.uniform([], minval, maxval, dtype=tf.int32)

        return tf.map_fn(
            _sample_single,
            (minimums, upper_bounds),
            dtype=tf.int32
        )

    def _crop_feature_squares(
            self,
            features: tf.Tensor,
            crop_x: tf.Tensor,
            crop_y: tf.Tensor
    ) -> tf.Tensor:
        # TODO: I am 100% sure this is achievable via some clever striding or gather_nd.
        #  However, I implemented a naive (and potentially slower) version simply for time reasons.
        def _crop_single(inputs: typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
            sample, offset_x, offset_y = inputs
            return tf.image.crop_to_bounding_box(
                sample,
                offset_y,
                offset_x,
                self.feature_rectangle_size,
                self.feature_rectangle_size
            )

        return tf.map_fn(
            _crop_single,
            (features, crop_x, crop_y),
            dtype=tf.float32
        )


class FC2DHead(Base2DHead):
    # TODO: Document everything here

    def __init__(
            self,
            backbone: tf.keras.Model,
            features: int,
            feature_rectangle_size: int,
            undo_spatial_transformations: bool,
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super(FC2DHead, self).__init__(
            backbone,
            feature_rectangle_size,
            undo_spatial_transformations,
            **kwargs
        )

        # 1x1 convolution emulating the fully-connected layer per location
        # Note that the intermediate inputs are already flattened into a rank 3 tensor
        self.conv = tf.keras.layers.Conv1D(
            features,
            kernel_size=1,
            activation=None,
            use_bias=True,
            kernel_initializer=dense_initializer,  # Dense init since we effectively do that here
            kernel_regularizer=kernel_regularizer
        )

    def call_output(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Generate output features and normalize to unit norm
        unscaled_output_features = self.conv(inputs)
        output_features = tf.math.l2_normalize(unscaled_output_features, axis=-1)
        return output_features


class MLP2DHead(Base2DHead):
    # TODO: Document everything here

    def __init__(
            self,
            backbone: tf.keras.Model,
            output_features: int,
            intermediate_features: int,
            feature_rectangle_size: int,
            undo_spatial_transformations: bool,
            dense_initializer: typing.Union[str, tf.keras.initializers.Initializer] = 'he_uniform',
            kernel_regularizer: typing.Optional[tf.keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super(MLP2DHead, self).__init__(
            backbone,
            feature_rectangle_size,
            undo_spatial_transformations,
            **kwargs
        )

        # 1x1 convolution emulating a fully-connected layer per location,
        # transforming the representations in a space better suited for comparison.
        # Note that the intermediate inputs are already flattened into a rank 3 tensor.
        self.conv_intermediate = tf.keras.layers.Conv1D(
            intermediate_features,
            kernel_size=1,
            activation='relu',  # No normalization, thus ReLU can be performed as part of the layer
            use_bias=True,
            kernel_initializer=dense_initializer,
            kernel_regularizer=kernel_regularizer
        )

        # Output 1x1 convolution for the fully connected layer creating the features used in the inner product
        self.conv_out = tf.keras.layers.Conv1D(
            output_features,
            kernel_size=1,
            activation=None,
            use_bias=True,
            kernel_initializer=dense_initializer,
            kernel_regularizer=kernel_regularizer
        )

    def call_output(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        # Generate intermediate features from input
        intermediate_features = self.conv_intermediate(inputs)

        # Generate output features and normalize to unit norm
        unscaled_output_features = self.conv_out(intermediate_features)
        output_features = tf.math.l2_normalize(unscaled_output_features, axis=-1)
        return output_features
