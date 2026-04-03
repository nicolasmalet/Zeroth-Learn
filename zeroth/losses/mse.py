import numpy as np

from ..abstract.loss import Loss
from ..first_order.layer import Layer
from ..types import Array


class MSE(Loss):
    @staticmethod
    def compute_loss(Y_pred: Array, Y_true: Array) -> float:
        return float(np.mean((Y_pred - Y_true) ** 2))

    @staticmethod
    def compute_batch_losses(Y_pred: Array, Y_true: Array) -> Array:
        return np.mean((Y_pred - Y_true) ** 2, axis=1)

    @staticmethod
    def compute_perturbed_losses(pY_pred: Array, Y_true: Array) -> Array:
        return np.mean((pY_pred - Y_true) ** 2, axis=2)

    @staticmethod
    def compute_gradient_wrt_activation(Y_pred: Array, Y_true: Array) -> Array:
        return 2 * (Y_pred - Y_true) / Y_true.shape[1]

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer: Layer, Y_pred: Array, Y_true: Array) -> Array:
        dL_dA = 2 * (Y_pred - Y_true) / Y_true.shape[1]
        dL_dZ = dL_dA * last_layer.activation.derivative(last_layer.Z)
        return dL_dZ

    def compute_losses_for_zeroth_order(self, pY_pred: Array, Y_true: Array) -> tuple[float, Array]:
        return (self.compute_loss(pY_pred[0], Y_true),
                self.compute_perturbed_losses(pY_pred, Y_true))

    def compute_losses_for_first_order(self, last_layer, Y_pred, Y_true) -> tuple[float, Array]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
