import numpy as np

from ..abstract.loss import Loss
from ..first_order.layer import Layer
from ..types import Array


class CrossEntropy(Loss):
    name = "CrossEntropy"

    @staticmethod
    def compute_loss(Y_pred: Array, Y_true: Array) -> float:
        Y_true_flat = Y_true.squeeze()
        idx = np.arange(Y_pred.shape[0])
        return float(- np.mean(np.log(1e-15 + Y_pred[idx, Y_true_flat])))

    @staticmethod
    def compute_batch_losses(Y_pred: Array, Y_true: Array) -> Array:
        Y_true_flat = Y_true.squeeze()
        idx = np.arange(Y_pred.shape[0])
        return - np.log(1e-15 + Y_pred[idx, Y_true_flat])

    @staticmethod
    def compute_perturbed_losses(pY_pred: Array, Y_true: Array) -> Array:
        Y_true_flat = Y_true.squeeze()
        idx = np.arange(pY_pred.shape[1])
        return - np.log(1e-15 + pY_pred[:, idx, Y_true_flat])

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer: Layer, Y_pred: Array, Y_true: Array) -> Array:
        Y_true_flat = Y_true.squeeze()
        dZ = Y_pred.copy()
        batch_size = Y_pred.shape[0]
        dZ[np.arange(batch_size), Y_true_flat] -= 1.0
        return dZ

    def compute_losses_for_zeroth_order(self, pY_pred: Array, Y_true: Array) -> tuple[float, Array]:
        avg_loss = self.compute_loss(pY_pred[0], Y_true)
        p_loss = self.compute_perturbed_losses(pY_pred, Y_true)
        return avg_loss, p_loss

    def compute_losses_for_first_order(self, last_layer, Y_pred: Array, Y_true: Array) -> tuple[float, Array]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
