import sklearn
import sklearn.linear_model
import tensorflow as tf


def create_old_logreg_model(seed: int) -> sklearn.linear_model.LogisticRegression:
    return sklearn.linear_model.LogisticRegression(
        C=100000.0,
        class_weight='balanced',
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        multi_class='ovr',
        n_jobs=1,
        penalty='l2',
        random_state=seed,
        tol=0.0001,
        verbose=0,
        warm_start=False
    )


class BaselineCNN(tf.keras.Model):
    def __init__(self, dropout_rate: float):
        super(BaselineCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            padding='same',
            activation='relu',
        )
        self.dropout1 = tf.keras.layers.SpatialDropout2D(dropout_rate)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            padding='same',
            activation='relu'
        )
        self.dropout2 = tf.keras.layers.SpatialDropout2D(dropout_rate)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

        self.fcconv = tf.keras.layers.Conv2D(
            filters=256,
            kernel_size=(5, 5),
            strides=(4, 4),
            padding='same',
            activation='relu'
        )

        self.conv_out = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            padding='valid',
            activation=None
        )

    def call(self, inputs, training=None, mask=None):
        xs = self.conv1(inputs)
        xs = self.dropout1(xs)
        xs = self.pool1(xs)

        xs = self.conv2(xs)
        xs = self.dropout2(xs)
        xs = self.pool2(xs)

        xs = self.fcconv(xs)

        logits = self.conv_out(xs)
        return logits
