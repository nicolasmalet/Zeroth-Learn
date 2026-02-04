import numpy as np


class Data:
    """
    Manages dataset loading, shuffling, and batching.
    """

    def __init__(self, raw_X_train: np.ndarray, raw_Y_train: np.ndarray, raw_X_test: np.ndarray, raw_Y_test: np.ndarray):
        self.input_dim: int = raw_X_train.shape[0]
        self.output_dim: int = raw_Y_train.shape[0]
        self.nb_data: int = raw_X_train.shape[1]
        self.nb_tests: int = raw_X_test.shape[1]

        self.batch_size: int | None = None
        self.nb_batches: int | None = None

        self.raw_X_train: np.ndarray = raw_X_train
        self.raw_Y_train: np.ndarray = raw_Y_train
        self.X_test: np.ndarray = raw_X_test
        self.Y_test: np.ndarray = raw_Y_test

        self.X_train: np.ndarray = np.array([])
        self.Y_train: np.ndarray = np.array([])

    def prepare_data(self, batch_size):
        """Shuffles and splits the raw data into batches for a new epoch."""
        self.batch_size = batch_size
        self.nb_batches = self.nb_data // self.batch_size

        indexes = np.random.permutation(self.nb_data)

        # Shuffle raw data
        raw_X_train = self.raw_X_train[:, indexes]
        raw_Y_train = self.raw_Y_train[:, indexes]

        # Split into batches (Shape: nb_batches, dim, batch_size)
        self.X_train = np.stack(np.split(raw_X_train, self.nb_batches, axis=1), axis=0)
        self.Y_train = np.stack(np.split(raw_Y_train, self.nb_batches, axis=1), axis=0)