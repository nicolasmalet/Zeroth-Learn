import numpy as np

from ..abstract.activation import Activation
from ..types import Array


class ReLU(Activation):
    def __call__(self, x: float | Array) -> float | Array:
        return np.maximum(x, 0)

    def derivative(self, x: float | Array) -> float | Array:
        return np.heaviside(x, 0).astype(int)


class Sigmoid(Activation):
    def __call__(self, x: float | Array) -> float | Array:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: float | Array) -> float | Array:
        s = self.__call__(x)
        return s * (1 - s)


class Identity(Activation):
    def __call__(self, x: float | Array) -> float | Array:
        return x

    def derivative(self, x: float | Array) -> float | Array:
        return np.ones_like(x)


class Softmax(Activation):
    def __call__(self, x: float | Array) -> float | Array:
        """
        Stabilization: We subtract the maximum to avoid overflow (inf)
        axis=-1 is important for handling:
        - (batch, output_dim) -> standard calculation
        - (T, batch, output_dim) -> calculation for zeroth_order
        """
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(shift_x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def derivative(self, x: float | Array) -> float | Array:
        raise NotImplementedError("no need for derivative when using CrossEntropy loss")
