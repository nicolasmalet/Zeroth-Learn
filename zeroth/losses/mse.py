from ..first_order.layer import Layer
from ..abstract.loss import Loss


import numpy as np


class MSE(Loss):
    name = "MSE"

    @staticmethod
    def compute_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        return np.mean((Y_pred - Y_true) ** 2, axis=(0, 1))

    @staticmethod
    def compute_batch_losses(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return np.mean((Y_pred - Y_true) ** 2, axis=0)

    @staticmethod
    def compute_perturbed_losses(pY_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return np.mean((pY_pred - Y_true) ** 2, axis=1)

    @staticmethod
    def compute_gradient_wrt_activation(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        return 2 * np.mean(Y_pred - Y_true, axis=0)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer: Layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        dL_dA = 2 * (Y_pred - Y_true) / Y_true.size
        dL_dZ = dL_dA * last_layer.df(last_layer.Z)
        return dL_dZ

    def compute_losses_for_zeroth_order(self, pY_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        return (self.compute_loss(pY_pred[0], Y_true),
                self.compute_perturbed_losses(pY_pred, Y_true))

    def compute_losses_for_first_order(self, last_layer, Y_pred, Y_true) -> tuple[float, np.ndarray]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
