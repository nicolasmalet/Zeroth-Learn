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
    def compute_losses_for_first_order(self, last_layer, Y_pred: np.ndarray, Y_true: np.ndarray):
        pass


class MSE(Loss):
    name = "MSE"

    @staticmethod
    def compute_loss(Y_pred, Y_true) -> float:
        return np.mean((Y_pred - Y_true) ** 2, axis=(0, 1))

    @staticmethod
    def compute_batch_losses(Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2, axis=0)

    @staticmethod
    def compute_perturbed_losses(pY_pred, Y_true):
        return np.mean((pY_pred - Y_true) ** 2, axis=1)

    @staticmethod
    def compute_gradient_wrt_activation(Y_pred, Y_true):
        return 2 * np.mean(Y_pred - Y_true, axis=0)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true):
        dL_dA = 2 * (Y_pred - Y_true) / Y_true.size
        dL_dZ = dL_dA * last_layer.df(last_layer.Z)
        return dL_dZ

    def compute_losses_for_zeroth_order(self, pY_pred, Y_true):
        return (self.compute_loss(pY_pred[0], Y_true),
                self.compute_perturbed_losses(pY_pred, Y_true))

    def compute_losses_for_first_order(self, last_layer, Y_pred, Y_true):
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)


class CrossEntropy(Loss):
    name = "CrossEntropy"

    @staticmethod
    def compute_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        idx = np.arange(Y_pred.shape[1])
        return - np.mean(np.log(1e-8 + Y_pred[Y_true, idx]))

    @staticmethod
    def compute_batch_losses(Y_pred: np.ndarray, Y_true: np.ndarray):
        idx = np.arange(Y_pred.shape[1])
        return - np.log(1e-8 + Y_pred[Y_true, idx])

    @staticmethod
    def compute_perturbed_losses(pY_pred: np.ndarray, Y_true: np.ndarray):
        idx = np.arange(pY_pred.shape[2])
        return - np.mean(np.log(1e-8 + pY_pred[:, Y_true, idx]), axis=1)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred: np.ndarray, Y_true: np.ndarray):
        dZ = Y_pred.copy()
        batch_size = Y_pred.shape[1]
        dZ[Y_true[0], np.arange(batch_size)] -= 1.0
        return dZ

    def compute_losses_for_zeroth_order(self, pY_pred: np.ndarray, Y_true: np.ndarray):
        avg_loss = self.compute_loss(pY_pred[0], Y_true)
        p_loss = self.compute_perturbed_losses(pY_pred, Y_true)
        return avg_loss, p_loss

    def compute_losses_for_first_order(self, last_layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
