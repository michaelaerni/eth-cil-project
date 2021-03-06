import abc

import tensorflow as tf


class _BinaryThresholdMeanMetric(tf.keras.metrics.Metric, metaclass=abc.ABCMeta):
    def __init__(self, name: str, threshold: float = 0.5, model_output_stride: int = 1, dtype=None):
        super(_BinaryThresholdMeanMetric, self).__init__(name=name, dtype=dtype)
        self._threshold = threshold
        self._model_output_stride = model_output_stride

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

    @abc.abstractmethod
    def score_batch(
            self,
            y_true: tf.Tensor,
            y_pred: tf.Tensor,
            sample_weight=None
    ) -> tf.Tensor:
        """
        Calculate the metric of interest over a batch of binary predictions.

        Args:
            y_true: True labels of shape (batch_size, num_labels), either 0 or 1.
            y_pred: Predicted labels of shape (batch_size, num_labels), either 0 or 1.
            sample_weight: Sample weights.

        Returns:
            1D tensor of shape (batch_size,) containing the scores per sample.
        """
        pass

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Cast and threshold labels
        y_true = tf.cast(y_true > self._threshold, self._dtype)
        y_pred = tf.cast(y_pred > self._threshold, self._dtype)

        y_true = self._downsample_cil_threshold(y_true)
        y_pred = self._downsample_cil_threshold(y_pred)

        # Fix input shapes to be (batch size, num predictions)
        y_true = self._fix_shapes(y_true)
        y_pred = self._fix_shapes(y_pred)

        # First dimension is assumed to be batch
        batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)

        scores = self.score_batch(y_true, y_pred, sample_weight)

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

    def _downsample_cil_threshold(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Uses average pooling to down sample to stride 16, then applies threshold 0.25 to values.
        This function assumes that the values of the inputs tensor are in [0, 1].
        Args:
            inputs: Input tensor with values in [0, 1].

        Returns:
            Down sampled tensor with applied threshold.

        """

        stride = 16//self._model_output_stride
        threshold = 0.25

        expanded = False
        if len(tf.shape(inputs)) == 3:
            expanded = True
            inputs = tf.expand_dims(inputs, axis=0)

        # Have to transpose before applying average pooling,
        # as otherwise we get an error from the TensorFlow layout optimizer.
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        downsampled = tf.nn.avg_pool2d(
            inputs,
            ksize=(stride, stride),
            strides=(stride, stride),
            padding='SAME',
            data_format='NCHW'
        )
        downsampled = tf.transpose(downsampled, [0, 2, 3, 1])

        thresholded = tf.where(downsampled >= threshold, 1.0, 0.0)

        if expanded:
            thresholded = tf.squeeze(thresholded, axis=0)

        return thresholded


class BinaryMeanIoUScore(_BinaryThresholdMeanMetric):
    """
    Score for the mean IoU (intersection over union) for binary classification problems
    where the IoU is calculated per sample.
    The mean is then calculated over all observed samples until reset.

    This score accepts continuous inputs where the class membership (0 or 1)
    is determined via a threshold.
    """

    def __init__(self, name: str = 'binary_mean_iou_score', threshold: float = 0.5, model_output_stride: int = 1, dtype=None):
        super(BinaryMeanIoUScore, self).__init__(name=name, threshold=threshold, model_output_stride=model_output_stride, dtype=dtype)

    def score_batch(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        true_positives = tf.reduce_sum(
            tf.cast((y_pred == y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_predictions = tf.reduce_sum(
            tf.cast(y_pred != y_true, self._dtype),
            axis=1
        )

        # Calculate batch-wise score
        return tf.math.divide_no_nan(
            true_positives,
            true_positives + false_predictions
        )


class BinaryMeanFScore(_BinaryThresholdMeanMetric):
    """
    Mean F1 score for binary classification problems
    where the score is calculated per sample.
    The mean is then calculated over all observed samples until reset.

    This score accepts continuous inputs where the class membership (0 or 1)
    is determined via a threshold.
    """

    def __init__(self, name: str = 'binary_mean_f_score', threshold: float = 0.5, model_output_stride: int = 1, dtype=None):
        super(BinaryMeanFScore, self).__init__(name=name, threshold=threshold, model_output_stride=model_output_stride, dtype=dtype)

    def score_batch(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        true_positives = tf.reduce_sum(
            tf.cast((y_pred == y_true) & (y_pred == 1), self._dtype),
            axis=1
        )
        false_predictions = tf.reduce_sum(
            tf.cast(y_pred != y_true, self._dtype),
            axis=1
        )

        # Calculate batch-wise score
        return tf.math.divide_no_nan(
            2.0 * true_positives,
            2.0 * true_positives + false_predictions
        )


class BinaryMeanAccuracyScore(_BinaryThresholdMeanMetric):
    """
    Mean accuracy score for binary classification problems
    where the score is calculated per sample.
    The mean is then calculated over all observed samples until reset.

    This score accepts continuous inputs where the class membership (0 or 1)
    is determined via a threshold.
    """

    def __init__(self, name: str = 'binary_mean_accuracy', threshold: float = 0.5, model_output_stride: int = 1, dtype=None):
        super(BinaryMeanAccuracyScore, self).__init__(name=name, threshold=threshold, model_output_stride=model_output_stride, dtype=dtype)

    def score_batch(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight=None) -> tf.Tensor:
        # Return mean number of true predictions (= accuracy) per sample
        return tf.reduce_mean(
            tf.cast(y_pred == y_true, self._dtype),
            axis=1
        )
