from abc import ABC, abstractmethod

from ..types import Array


class Loss(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @staticmethod
    @abstractmethod
    def compute_loss(Y_pred: Array, Y_true: Array) -> float:
        """
        :param Y_pred: shape (batch, out)
        :param Y_true: shape (batch, out)
        :return: avg loss shape: float
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_batch_losses(Y_pred: Array, Y_true: Array) -> Array:
        """
        :param Y_pred: shape (batch, out)
        :param Y_true: shape (batch, out)
        :return: batch loss shape (batch, )
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_perturbed_losses(pY_pred: Array, Y_true: Array) -> Array:
        """
        :param pY_pred: (T, batch, out)
        :param Y_true: (batch, out)
        :return: perturbed loss (T, nb_params)
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred: Array, Y_true: Array) -> Array:
        """
        :param last_layer: last_layer of the network
        :param Y_pred: shape (batch, out)
        :param Y_true: shape (batch, out)
        :return: batch loss shape (batch, )
        """
        pass

    @abstractmethod
    def compute_losses_for_zeroth_order(self, pY_pred: Array, Y_true: Array) -> tuple[float, Array]:
        pass

    @abstractmethod
    def compute_losses_for_first_order(self, last_layer, Y_pred: Array, Y_true: Array) -> tuple[
        float, Array]:
        pass
