import numpy as np


def accuracy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return np.sum(np.argmax(Y_pred, axis=1) == Y_true[:, 0]).item() / Y_true.shape[0]
