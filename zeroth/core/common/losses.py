from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    @staticmethod
    @abstractmethod
    def get_avg_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: avg loss shape: float
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_batch_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_p_loss(pY_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param pY_pred: (T, out, batch)
        :param Y_true: (out, batch)
        :return: perturbed loss (T, )
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_d_loss(Y_pred: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
        """
        :param Y_pred: shape (out, batch)
        :param Y_true: shape (out, batch)
        :return: batch loss shape (batch, )
        """
        raise NotImplementedError


class MSE(Loss):
    name = "MSE"

    @staticmethod
    def get_avg_loss(Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2, axis=(0, 1))

    @staticmethod
    def get_batch_loss(Y_pred, Y_true):
        return np.mean((Y_pred - Y_true) ** 2, axis=0)

    @staticmethod
    def get_p_loss(pY_pred, Y_true):
        return np.mean((pY_pred - Y_true) ** 2, axis=1)

    @staticmethod
    def get_d_loss(Y_pred, Y_true):
        return 2 * np.mean(Y_pred - Y_true, axis=0)

    def get_losses_for_perturbation(self, Y_pred, pY_pred, data, batch_idx):
        Y_true = data.Y_train[batch_idx]
        return (self.get_avg_loss(Y_pred, Y_true),
                self.get_batch_loss(Y_pred, Y_true),
                self.get_p_loss(pY_pred, Y_true))

    @staticmethod
    def get_final_dZ(last_layer, Y_pred, Y_true):
        dL_dA = 2 * (Y_pred - Y_true) / Y_true.size
        dL_dZ = dL_dA * last_layer.df(last_layer.Z)
        return dL_dZ

    def get_backprop_init(self, last_layer, Y_pred, data, batch_index):

        Y_true = data.Y_train[batch_index]
        avg_loss = self.get_avg_loss(Y_pred, Y_true)
        dL_dZ = self.get_final_dZ(last_layer, Y_pred, Y_true)  #
        return avg_loss, dL_dZ


class CrossEntropy(Loss):
    name = "CrossEntropy"

    @staticmethod
    def get_avg_loss(Y_pred, Y_true):
        idx = np.arange(Y_pred.shape[1])
        return - np.mean(np.log(1e-8 + Y_pred[Y_true, idx]))

    @staticmethod
    def get_batch_loss(Y_pred, Y_true):
        idx = np.arange(Y_pred.shape[1])
        return - np.log(1e-8 + Y_pred[Y_true, idx])

    @staticmethod
    def get_p_loss(pY_pred, Y_true):
        idx = np.arange(pY_pred.shape[2])
        return - np.mean(np.log(1e-8 + pY_pred[:, Y_true, idx]), axis=1)

    @staticmethod
    def get_d_loss(Y_pred, Y_true):
        dZ = Y_pred.copy()
        batch_size = Y_pred.shape[1]
        dZ[Y_true[0], np.arange(batch_size)] -= 1.0
        return dZ

    def get_losses_for_perturbation(self, Y_pred, pY_pred, data, batch_idx):
        Y_true = data.Y_train[batch_idx]
        batch_loss = self.get_batch_loss(Y_pred, Y_true)
        avg_loss = np.mean(batch_loss)
        p_loss = self.get_p_loss(pY_pred, Y_true)
        return avg_loss, batch_loss, p_loss

    def get_loss_for_backpropagation(self, Y_pred, data, batch_idx):
        Y_true = data.Y_train[batch_idx]
        return self.get_avg_loss(Y_pred, Y_true), self.get_d_loss(Y_pred, Y_true)

    @staticmethod
    def get_final_dZ(Y_pred, Y_true):
        return Y_pred - Y_true

    def get_backprop_init(self, last_layer, Y_pred, data, batch_index):
        Y_true = data.Y_train[batch_index]
        avg_loss = self.get_avg_loss(Y_pred, Y_true)
        dL_dZ = self.get_d_loss(Y_pred, Y_true)  #
        return avg_loss, dL_dZ
