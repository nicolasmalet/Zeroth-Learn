import numpy as np
from sklearn.datasets import fetch_openml

from zeroth.abstract.data_creator import DataCreator
from zeroth.data import Data


class DataCreatorMnist(DataCreator):
    task_name: str = "mnist"
    def __call__(self) -> Data:
        mnist = fetch_openml('mnist_784', version=1)

        # Load data
        X = np.array(mnist.data.astype("float64"))  # 70 000 images
        Y = np.array(mnist.target.astype("int64"), dtype=int).reshape(-1, 1)

        X_train, X_test = X[:60000, :], X[60000:, :]
        Y_train, Y_test = Y[:60000, :], Y[60000:, :]

        # Normalize data
        mean = np.mean(X_train)
        std = np.std(X_train)
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        data_mnist = Data(X_train, Y_train, X_test, Y_test, 10)

        return data_mnist
