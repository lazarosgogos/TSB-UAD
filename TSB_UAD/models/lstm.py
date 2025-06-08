from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.layers import LSTM
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import Adam
from tensorflow import config, debugging
import tensorflow as tf

print("Num GPUs Available: ", len(config.list_physical_devices("GPU")))
# debugging.set_log_device_placement(True)

import numpy as np
import math
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler


class lstm:
    """
    Implementation of LSTM-AD

    Parameters
    ----------
    slidingwindow : int
        Subsequence length to analyze.
    predict_time_steps : int, (default=1)
        The length of the subsequence to predict.
    epochs : int, (default=10)
        Number of epochs for the training phase
    patience : int, (default=10)
        Number of epoch to wait before early stopping during training

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples - subsequence_length,)
        The anomaly score.
        The higher, the more abnormal. Anomalies tend to have higher
        scores. This value is available once decision_function is called.
    """

    def __init__(
        self, slidingwindow=100, predict_time_steps=1, epochs=10, patience=10, verbose=1
    ):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.batch_size = 64
        self.model_name = "LSTM2"

    def fit(self, X_clean, X_dirty, ratio=0.15):
        """Fit detector.

        Parameters
        ----------
        X_clean : numpy array of shape (n_samples, )
            The input training samples.
        X_dirty : numpy array of shape (n_samples, )
            The input testing samples.
        ratio : float, ([0,1])
            The ratio for the train validation split

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.n_test_ = len(X_dirty)

        split_index = int(len(X_clean) * (1 - ratio))

        # Training
        train_ds = X_clean[:split_index]
        val_ds = X_clean[split_index:]

        train_ds = self.create_dataset(
            train_ds, self.slidingwindow, self.predict_time_steps
        )
        val_ds = self.create_dataset(
            val_ds, self.slidingwindow, self.predict_time_steps
        )

        # Test
        test_ds = self.create_dataset(
            X_dirty, self.slidingwindow, self.predict_time_steps
        )

        for x, y in train_ds.take(1):
            print("x shape:", x.shape)  # (batch_size, slidingwindow, 1)
            print("y shape:", y.shape)  # (batch_size, predict_time_steps)

        # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        # X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.slidingwindow, 1)))
        model.add(LSTM(50))
        model.add(Dense(self.predict_time_steps))
        # model.compile(loss="mse", optimizer=Adam(learning_rate=1e-5, clipnorm=1.0))
        model.compile(loss="mse", optimizer="adam")
        print(model.summary())

        es = EarlyStopping(
            monitor="val_loss", mode="min", verbose=self.verbose, patience=self.patience
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[es],
        )

        prediction = model.predict(test_ds)

        # Extract Y_test from test_ds:
        Y_test = []
        for _, y in test_ds.unbatch():
            Y_test.append(y.numpy())
        Y_test = np.array(Y_test)

        self.Y = Y_test
        self.estimation = prediction
        self.estimator = model
        self.n_initial = split_index

        return self

    def create_dataset(self, X, slidingwindow, predict_time_steps=1):
        total_window = slidingwindow + predict_time_steps
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.window(size=total_window, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(total_window))

        def normalize_and_split(window):
            # Reshape to 2D so axis=0 is the time axis (like sklearn)
            window = tf.reshape(window, [-1, 1])  # shape (window_len, 1)

            min_val = tf.reduce_min(window, axis=0)
            max_val = tf.reduce_max(window, axis=0)

            scale = tf.maximum(max_val - min_val, 1e-4)
            scaled = (window - min_val) / scale
            scaled = scaled * (max_val - min_val) + min_val
            scaled = tf.clip_by_value(
                scaled, 0.0, 1.0
            )  # optional, but mimics sklearn range safety

            # Split X/y
            X_window = scaled[:slidingwindow]  # shape (slidingwindow, 1)
            y_window = tf.reshape(
                scaled[slidingwindow:], [-1]
            )  # shape (predict_steps,)
            return X_window, y_window

        # def normalize_and_split(window):
        #     return tf.data.Dataset.from_tensor_slices(
        #         MinMaxScaler(feature_range=(0, 1))
        #         .fit_transform(window.unbatch().as_numpy_iterator())
        #         .ravel()
        #     )

        ds = ds.map(normalize_and_split, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds

    def decision_function(self, measure=None):
        """Derive the decision score based on the given distance measure

        Parameters
        ----------
        measure : object
            object for given distance measure with methods to derive the score

        Returns
        -------
        self : object
            Fitted estimator.
        """

        Y_test = self.Y

        score = np.zeros(self.n_test_)
        estimation = self.estimation

        for i in range(estimation.shape[0]):
            score[i - estimation.shape[0]] = measure.measure(
                Y_test[i], estimation[i], self.n_test_ - estimation.shape[0] + i
            )

        score[0 : -estimation.shape[0]] = score[-estimation.shape[0]]

        self.decision_scores_ = score
        return self
