from typing import Iterator

import numpy as np

from .types import Array


class Data:
    def __init__(self, raw_X_train: Array, raw_Y_train: Array, raw_X_test: Array, raw_Y_test: Array) -> None:
        self.raw_X_train: Array = raw_X_train
        self.raw_Y_train: Array = raw_Y_train
        self.X_test: Array = raw_X_test
        self.Y_test: Array = raw_Y_test

        self.nb_data: int = raw_X_train.shape[0]
        self.batch_size: int | None = None

        self.indices: np.ndarray = np.arange(self.nb_data)

    def __len__(self) -> int:
        return self.nb_data // self.batch_size

    def __iter__(self) -> Iterator[tuple[Array, Array]]:
        for i in range(0, self.nb_data - self.batch_size + 1, self.batch_size):
            batch_idx = self.indices[i:i + self.batch_size]
            yield self.raw_X_train[batch_idx], self.raw_Y_train[batch_idx]

    def permutation(self) -> None:
        self.indices = np.random.permutation(self.nb_data)
