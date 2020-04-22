import tensorflow as tf


class BinaryMeanIoUScore(tf.keras.metrics.Metric):
    def __init__(self, name: str = 'binary_iou_score', threshold: float = 0.5, dtype=None):
        super(BinaryMeanIoUScore, self).__init__(name=name, dtype=dtype)
        self._threshold = threshold

        # Add a variable to collection confusion matrices
        self._score_accumulator = self.add_weight(
            name='score_accumulator',
            shape=(),
            initializer=tf.initializers.zeros,
            dtype=self._dtype
        )
        self._sample_counter = self.add_weight(
            name='sample_counter',
            shape=(),
            initializer=tf.initializers.zeros,
            dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast and threshold labels
        y_true = tf.cast(y_true > self._threshold, self._dtype)
        y_pred = tf.cast(y_pred > self._threshold, self._dtype)

        # Fix input shapes to be (batch size, num predictions)
        y_true = self._fix_shapes(y_true)
        y_pred = self._fix_shapes(y_pred)

        # First dimension is assumed to be batch
        batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)

        true_positives = tf.reduce_sum(
            tf.cast((y_pred == y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_positives = tf.reduce_sum(
            tf.cast((y_pred != y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_negatives = tf.reduce_sum(
            tf.cast((y_pred != y_true) & (y_pred == 0), self._dtype),
            axis=1
        )

        # Calculate batch-wise score
        scores = tf.math.divide_no_nan(
            true_positives,
            true_positives + false_positives + false_negatives
        )

        # Update counter with observed number of samples
        self._sample_counter.assign_add(batch_size)

        # Finally update sum accumulator
        return self._score_accumulator.assign_add(
            tf.reduce_sum(scores)
        )

    def result(self):
        # Safely handle cases with zero observed samples
        result = tf.math.divide_no_nan(
            self._score_accumulator,
            tf.cast(self._sample_counter, self._dtype)
        )

        return result

    def reset_states(self):
        # Reset score sum and sample counter to zero
        tf.keras.backend.set_value(self._score_accumulator, 0.0)
        tf.keras.backend.set_value(self._sample_counter, 0)

    @classmethod
    def _fix_shapes(cls, values: tf.Tensor) -> tf.Tensor:
        if values.shape.ndims == 1:
            # Expand dimensions if non-batched
            # Assumes values is are the predictions for a single sample
            values = tf.expand_dims(values, axis=0)

        if values.shape.ndims > 2:
            # Flatten samples
            # Assumes first dimension is batch size, rest is predictions
            values = tf.reshape(values, (tf.shape(values)[0], -1))

        return values


class BinaryMeanFScore(tf.keras.metrics.Metric):
    def __init__(self, name: str = 'binary_mean_f_score', threshold: float = 0.5, dtype=None):
        super(BinaryMeanFScore, self).__init__(name=name, dtype=dtype)
        self._threshold = threshold

        # Add a variable to collection confusion matrices
        self._score_accumulator = self.add_weight(
            name='score_accumulator',
            shape=(),
            initializer=tf.initializers.zeros,
            dtype=self._dtype
        )
        self._sample_counter = self.add_weight(
            name='sample_counter',
            shape=(),
            initializer=tf.initializers.zeros,
            dtype=tf.int64
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast and threshold labels
        y_true = tf.cast(y_true > self._threshold, self._dtype)
        y_pred = tf.cast(y_pred > self._threshold, self._dtype)

        # Fix input shapes to be (batch size, num predictions)
        y_true = self._fix_shapes(y_true)
        y_pred = self._fix_shapes(y_pred)

        # First dimension is assumed to be batch
        batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)

        true_positives = tf.reduce_sum(
            tf.cast((y_pred == y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_positives = tf.reduce_sum(
            tf.cast((y_pred != y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_negatives = tf.reduce_sum(
            tf.cast((y_pred != y_true) & (y_pred == 0), self._dtype),
            axis=1
        )

        # Calculate batch-wise score
        scores = tf.math.divide_no_nan(
            2.0 * true_positives,
            2.0 * true_positives + false_positives + false_negatives
        )

        # Update counter with observed number of samples
        self._sample_counter.assign_add(batch_size)

        # Finally update sum accumulator
        return self._score_accumulator.assign_add(
            tf.reduce_sum(scores)
        )

    def result(self):
        # Safely handle cases with zero observed samples
        result = tf.math.divide_no_nan(
            self._score_accumulator,
            tf.cast(self._sample_counter, self._dtype)
        )

        return result

    def reset_states(self):
        # Reset score sum and sample counter to zero
        tf.keras.backend.set_value(self._score_accumulator, 0.0)
        tf.keras.backend.set_value(self._sample_counter, 0)

    @classmethod
    def _fix_shapes(cls, values: tf.Tensor) -> tf.Tensor:
        if values.shape.ndims == 1:
            # Expand dimensions if non-batched
            # Assumes values is are the predictions for a single sample
            values = tf.expand_dims(values, axis=0)

        if values.shape.ndims > 2:
            # Flatten samples
            # Assumes first dimension is batch size, rest is predictions
            values = tf.reshape(values, (tf.shape(values)[0], -1))

        return values
