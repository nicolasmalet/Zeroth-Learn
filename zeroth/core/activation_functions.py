import numpy as np


def relu(x):
    return np.maximum(x, 0)


def heaviside(x):
    return np.heaviside(x, 0).astype(int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Stabilization: We subtract the maximum to avoid overflow (inf)
    axis=-2 is important for handling:
    - (output_dim, batch) -> standard calculation
    - (T, output_dim, batch) -> calculation for spsa
    """
    shift_x = x - np.max(x, axis=-2, keepdims=True)
    e = np.exp(shift_x)
    return e / np.sum(e, axis=-2, keepdims=True)


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def identity(x):
    return x


get_df = {
    relu: heaviside,
    sigmoid: sigmoid_derivative,
    identity: np.ones_like,
    softmax: None  # No need to as the CrossEntropy class implements the differentiation
}
