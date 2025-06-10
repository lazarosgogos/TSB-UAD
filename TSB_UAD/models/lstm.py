from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM
from keras._tf_keras.keras.callbacks import EarlyStopping
from keras._tf_keras.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from collections import deque

from TSB_UAD.models.feature import Window


class lstm:
    """
    LSTM-based anomaly detector with optional streaming support.

    Parameters
    ----------
    slidingwindow : int
        Number of time steps for the input sequence.
    predict_time_steps : int
        Number of time steps to predict.
    epochs : int
        Number of training epochs.
    patience : int
        Early stopping patience.
    streaming : bool
        Whether to enable streaming (online) mode.
    verbose : int
        Verbosity level.
    buffer_size : int
        Maximum number of recent observations to keep in streaming mode.
    error_window : int
        Number of past reconstruction errors used for z-score calculation.
    retrain_z_thresh : float
        Z-score threshold above which retraining is triggered in streaming mode.

    Attributes
    ----------
    model : keras.Model
        The trained LSTM model.
    reconstruction_errors : deque
        Rolling window of recent reconstruction errors.
    buffer : deque
        Data buffer for streaming mode.
    decision_scores_ : np.ndarray
        Computed anomaly scores.
    """

    def __init__(
        self,
        slidingwindow=100,
        predict_time_steps=1,
        epochs=10,
        patience=10,
        streaming=False,
        verbose=1,
        buffer_size=500,
        error_window=100,
        retrain_z_thresh=3.0,
    ):
        self.slidingwindow = slidingwindow
        self.predict_time_steps = predict_time_steps
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.streaming = streaming
        self.batch_size = 64

        self.model = None
        self.reconstruction_errors = deque(maxlen=error_window)
        self.bufferX = deque(maxlen=buffer_size)
        self.bufferY = deque(maxlen=buffer_size)
        self.retrain_z_thresh = retrain_z_thresh
        self.n_test_ = None
        self.Y = None
        self.estimation = None

        self.mu = 0.0
        self.std = 1.0

    def _create_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.slidingwindow, 1)))
        model.add(LSTM(50))
        model.add(Dense(self.predict_time_steps))
        model.compile(loss="mse", optimizer="adam")
        return model

    def _create_dataset(self, X):
        """
        Create a TensorFlow dataset with normalized sliding windows.

        Parameters
        ----------
        X : array-like
            Time series input data.

        Returns
        -------
        tf.data.Dataset
            Prepared dataset of (X, y) pairs.
        """
        total_window = self.slidingwindow + self.predict_time_steps
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        print(f"{X.get_shape()}")
        print(f"{X=}")

        ds = tf.data.Dataset.from_tensor_slices(X)
        ds = ds.window(size=total_window, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(total_window))

        def normalize_and_split(window):
            window = tf.reshape(window, [-1, 1])
            min_val = tf.reduce_min(window)
            max_val = tf.reduce_max(window)
            scale = tf.maximum(max_val - min_val, 1e-4)
            scaled = (window - min_val) / scale
            X_window = scaled[: self.slidingwindow]
            y_window = tf.reshape(scaled[self.slidingwindow :], [-1])
            return X_window, y_window

        ds = ds.map(normalize_and_split, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def fit(self, X_clean, X_dirty=None, ratio=0.15):
        """
        Train the model using clean training data.

        Parameters
        ----------
        X_clean : array-like
            Clean training data.
        X_dirty : array-like or None
            Optional separate test data. Defaults to X_clean.
        ratio : float
            Ratio of validation data to hold out from X_clean.
        """
        if self.streaming:
            raise ValueError("Use partial_fit in streaming mode")
        
        if X_dirty is None:
            X_dirty = X_clean
            X_clean = X_clean[: int(0.1 * len(X_clean))]

        split_index = int(len(X_clean) * (1 - ratio))
        train, val = X_clean[:split_index], X_clean[split_index:]
        train_ds = self._create_dataset(train)
        val_ds = self._create_dataset(val)

        self.model = self._create_model()
        es = EarlyStopping(
            monitor="val_loss", patience=self.patience, verbose=self.verbose
        )

        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[es],
        )

        test_ds = self._create_dataset(X_dirty)
        prediction = self.model.predict(test_ds)

        Y_test = []
        for _, y in test_ds.unbatch():
            Y_test.append(y.numpy())
        Y_test = np.array(Y_test)

        self.Y = Y_test
        self.estimation = prediction
        self.n_test_ = len(X_dirty)

        self.mu = np.mean(train)
        self.std = np.std(train)

    def partial_fit(self, x, y):
        """
        Update model with a single observation in streaming mode.

        Parameters
        ----------
        x : float
            New incoming observation.

        Returns
        -------
        float
            Z-score if enough error history exists, else 0.0.
        """
        if not self.streaming:
            raise ValueError("Use fit() in batch mode")

        self.bufferX.append(x)
        self.bufferY.append(y)
        if len(self.bufferX) < self.slidingwindow + self.predict_time_steps:
            return None

        a_x = np.array(self.bufferX)
        a_y = np.array(self.bufferY)

        # cond = a_x.mean() > self.mu + np.abs(self.std * 3)
        cond = np.abs(a_x.mean() - self.mu) > self.std * 3
        if cond:
            print("COND TRUE")
            self.mu = a_x.mean()
            self.std = a_x.std()

            buffer_dataset = self._create_dataset(np.column_stack((a_x, a_y)))

            if self.model is None:
                self.model = self._create_model()
                self._retrain_model(buffer_dataset)
                print(f"Distribution shift detected, retraining model")

        # window = list(self.buffer)[-self.slidingwindow - self.predict_time_steps :]
        # X_seq = np.array(window[: self.slidingwindow], dtype=np.float32)[:, 0].reshape(1, self.slidingwindow, 1)
        # print(f"{X_seq=}")

        # y_true = np.array(window[: self.slidingwindow], dtype=np.int32)[:, 1].reshape(1, -1)
        # print(f"{y_true=}")

        # X_seq = []
        # y_true = []
        # for x, y in dataset.unbatch():
        #     X_seq.append(x.numpy())
        #     y_true.append(y.numpy())
        # X_seq = np.array(X_seq)
        # y_true = np.array(y_true)

        # X_seq, y_true = list(dataset)
        # # y_true = np.array(list(tfds.as_numpy(dataset)))[:, 1]

        # y_pred = self.model.predict(X_seq, verbose=0)
        # error = np.mean(np.square(y_true - y_pred))
        # self.reconstruction_errors.append(error)

        # # Require a minimum number of past errors to ensure stable statistics
        # min_required_errors = 30  # Minimum errors to compute reliable z-score
        # if len(self.reconstruction_errors) >= min_required_errors:
        #     mean = np.mean(self.reconstruction_errors)
        #     std = np.std(self.reconstruction_errors)
        #     z = (error - mean) / std if std > 0 else 0
        #     if abs(z) > self.retrain_z_thresh:
        #         self._retrain_model()
        #     return z
        # return 0.0

    def _retrain_model(self, buffer_dataset):
        """
        Retrain model using current buffer contents.
        """
        print("Retraining model from buffer...")
        # X = np.array(self.bufferX, dtype=np.float32)
        # if len(X) < self.slidingwindow + self.predict_time_steps:
        #     return
        # dataset = self._create_dataset(X)

        es = EarlyStopping(
            monitor="val_loss", patience=self.patience, verbose=self.verbose
        )

        self.model.fit(
            buffer_dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[es],
        )
        prediction = self.model.predict(buffer_dataset)

        Y_test = []
        for _, y in buffer_dataset.unbatch():
            Y_test.append(y.numpy())
        Y_test = np.array(Y_test)

        self.Y = Y_test
        self.estimation = prediction

    def score_samples(self, X):
        """
        Compute reconstruction error for each window in X.

        Parameters
        ----------
        X : array-like
            Input time series data.

        Returns
        -------
        np.ndarray
            Reconstruction errors.
        """
        if self.model is None:
            raise RuntimeError("Model not trained")

        dataset = self._create_dataset(X)
        errors = []
        for X_batch, y_batch in dataset:
            y_pred = self.model.predict(X_batch, verbose=0)
            err = np.mean(np.square(y_batch - y_pred), axis=1)
            errors.extend(err)
        return np.array(errors)

    def decision_function(self, measure=None):
        """
        Derive decision scores using a provided distance measure.

        Parameters
        ----------
        measure : object
            Object implementing `measure(a, b, index)` function.

        Returns
        -------
        self
            The detector with `decision_scores_` populated.
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

    def setStreaming(self, enabled: bool):
        self.streaming = enabled
