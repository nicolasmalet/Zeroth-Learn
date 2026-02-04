from zeroth.core import Data

from sklearn.datasets import fetch_openml

import numpy as np


def create_data_mnist():
    mnist = fetch_openml('mnist_784', version=1)

    # Load data
    X = np.array(mnist.data.astype("float32")).T  # 70 000 images
    Y = np.array(mnist.target.astype("int64"), dtype=int).reshape(1, -1)

    X_train, X_test = X[:, :60000], X[:, 60000:]
    Y_train, Y_test = Y[:, :60000], Y[:, 60000:]

    # Normalize data
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    data_mnist = Data(X_train, Y_train, X_test, Y_test)

    return data_mnist
