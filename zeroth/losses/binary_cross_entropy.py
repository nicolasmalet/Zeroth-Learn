import numpy as np

from ..abstract.loss import Loss
from ..first_order.layer import Layer
from ..types import Array


class BinaryCrossEntropy(Loss):
    @staticmethod
    def compute_loss(Y_pred: Array, Y_true: Array) -> float:
        Y_true = Y_true.reshape(Y_pred.shape)
        loss = - (Y_true * np.log(Y_pred + 1e-15) + (1 - Y_true) * np.log(1 - Y_pred + 1e-15))
        return float(np.mean(loss))

    @staticmethod
    def compute_batch_losses(Y_pred: Array, Y_true: Array) -> Array:
        Y_true = Y_true.reshape(Y_pred.shape)
        loss = -(Y_true * np.log(Y_pred + 1e-15) + (1 - Y_true) * np.log(1 - Y_pred + 1e-15))
        return np.mean(loss, axis=1)

    @staticmethod
    def compute_perturbed_losses(pY_pred: Array, Y_true: Array) -> Array:
        Y_true = Y_true.reshape(1, pY_pred.shape[1], -1)
        loss = -(Y_true * np.log(pY_pred + 1e-15) + (1 - Y_true) * np.log(1 - pY_pred + 1e-15))
        return np.mean(loss, axis=2)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer: Layer, Y_pred: Array, Y_true: Array) -> Array:
        Y_true = Y_true.reshape(Y_pred.shape)
        return Y_pred - Y_true

    def compute_losses_for_zeroth_order(self, pY_pred: Array, Y_true: Array) -> tuple[float, Array]:
        avg_loss = self.compute_loss(pY_pred[0], Y_true)
        p_loss = self.compute_perturbed_losses(pY_pred, Y_true)
        return avg_loss, p_loss

    def compute_losses_for_first_order(self, last_layer: Layer, Y_pred: Array, Y_true: Array) -> tuple[float, Array]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
