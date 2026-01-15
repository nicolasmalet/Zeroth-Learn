import numpy as np


def ones(x):
    return np.ones_like(x)


def cross_entropy(Y_pred, Y_true):
    return -np.log(Y_pred[np.arange(Y_pred.shape[1]), Y_true])


def d_cross_entropy(Y_pred, Y_true):
    return -np.log(Y_pred[np.arange(Y_pred.shape[1]), Y_true])


def to_one_hot(M, n):
    p = M.shape[1]
    M_one_hot = np.zeros((n, p))
    M_one_hot[M, np.arange(p)] = 1
    return M_one_hot


def metric_mnist(Y_pred, Y_true):
    return np.sum(np.argmax(Y_pred, axis=0) == Y_true).item() / Y_true.shape[1]


def N2(Y_pred, Y_true):
    return np.mean((Y_true - Y_pred) ** 2)


def get_wrongs(Y_pred, Y_true):
    return np.argwhere(np.argmax(Y_pred, axis=0) != Y_true)[:, 1].flatten()


def get_goods(Y_pred, Y_true):
    return np.argwhere(np.argmax(Y_pred, axis=0) == Y_true)[:, 1].flatten()


def rademacher_matrix(k, n):
    S = np.random.choice([-1, 1], size=(k, n))
    P = S / np.sqrt(k)  # columns have exact L2 norm = 1
    return P


def get_name(value):
    if hasattr(value, "name"):
        return value.name
    return value


