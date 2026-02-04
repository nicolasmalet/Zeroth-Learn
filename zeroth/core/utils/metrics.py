import numpy as np


def accuracy(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return np.sum(np.argmax(Y_pred, axis=0) == Y_true).item() / Y_true.shape[1]
