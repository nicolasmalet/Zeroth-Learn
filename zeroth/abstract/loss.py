from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def compute_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: avg loss shape: float
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_batch_losses(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_perturbed_losses(pY_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param pY_pred: (T, out, batch)
        :param Y_true: (out, batch)
        :return: perturbed loss (nb_params, T)
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param last_layer: last_layer of the network
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        pass

    @abstractmethod
    def compute_losses_for_zeroth_order(self, pY_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        pass


    @abstractmethod
    def compute_losses_for_first_order(self, last_layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        pass