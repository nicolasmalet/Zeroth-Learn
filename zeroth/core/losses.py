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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_batch_losses(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_perturbed_losses(pY_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param pY_pred: (T, out, batch)
        :param Y_true: (out, batch)
        :return: perturbed loss (T, )
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param last_layer: last_layer of the network
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        raise NotImplementedError

    @abstractmethod
    def compute_losses_for_zeroth_order(self, Y_pred, pY_pred, data, batch_idx):
        raise NotImplementedError


    @abstractmethod
    def compute_losses_for_first_order(self, last_layer, Y_pred, data, batch_idx):
        raise NotImplementedError


class MSE(Loss):
    name = "MSE"

    @staticmethod
    def compute_loss(Y_pred, Y_true):
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

    def compute_losses_for_zeroth_order(self, Y_pred, pY_pred, data, batch_index):
        Y_true = data.Y_train[batch_index]
        return (self.compute_loss(Y_pred, Y_true),
                self.compute_batch_losses(Y_pred, Y_true),
                self.compute_perturbed_losses(pY_pred, Y_true))

    def compute_losses_for_first_order(self, last_layer, Y_pred, data, batch_index):
        Y_true = data.Y_train[batch_index]
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)


class CrossEntropy(Loss):
    name = "CrossEntropy"

    @staticmethod
    def compute_loss(Y_pred, Y_true):
        idx = np.arange(Y_pred.shape[1])
        return - np.mean(np.log(1e-8 + Y_pred[Y_true, idx]))

    @staticmethod
    def compute_batch_losses(Y_pred, Y_true):
        idx = np.arange(Y_pred.shape[1])
        return - np.log(1e-8 + Y_pred[Y_true, idx])

    @staticmethod
    def compute_perturbed_losses(pY_pred, Y_true):
        idx = np.arange(pY_pred.shape[2])
        return - np.mean(np.log(1e-8 + pY_pred[:, Y_true, idx]), axis=1)

    @staticmethod
    def compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true):
        dZ = Y_pred.copy()
        batch_size = Y_pred.shape[1]
        dZ[Y_true[0], np.arange(batch_size)] -= 1.0
        return dZ

    def compute_losses_for_zeroth_order(self, Y_pred, pY_pred, data, batch_idx):
        Y_true = data.Y_train[batch_idx]

        avg_loss = self.compute_loss(Y_pred, Y_true)
        batch_loss = self.compute_batch_losses(Y_pred, Y_true)
        p_loss = self.compute_perturbed_losses(pY_pred, Y_true)

        return avg_loss, batch_loss, p_loss

    def compute_losses_for_first_order(self, last_layer, Y_pred, data, batch_idx):
        Y_true = data.Y_train[batch_idx]
        return self.compute_loss(Y_pred, Y_true), self.compute_gradient_wrt_preactivation(last_layer, Y_pred, Y_true)
