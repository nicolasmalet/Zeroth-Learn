from ..first_order.layer import Layer
from ..abstract.loss import Loss

import numpy as np


class CrossEntropy(Loss):
    name = "CrossEntropy"

    @staticmethod
    def compute_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        idx = np.arange(Y_pred.shape[1])
        return - np.mean(np.log(1e-8 + Y_pred[Y_true, idx]))

    @staticmethod
    def compute_batch_losses(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        idx = np.arange(Y_pred.shape[1])
        return - np.log(1e-8 + Y_pred[Y_true, idx])

    @staticmethod
    def compute_perturbed_losses(pY_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        idx = np.arange(pY_pred.shape[2])
        return - np.mean(np.log(1e-8 + pY_pred[:, Y_true, idx]), axis=1)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer: Layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        dZ = Y_pred.copy()
        batch_size = Y_pred.shape[1]
        dZ[Y_true[0], np.arange(batch_size)] -= 1.0
        return dZ

    def compute_losses_for_zeroth_order(self, pY_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        avg_loss = self.compute_loss(pY_pred[0], Y_true)
        p_loss = self.compute_perturbed_losses(pY_pred, Y_true)
        return avg_loss, p_loss

    def compute_losses_for_first_order(self, last_layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
