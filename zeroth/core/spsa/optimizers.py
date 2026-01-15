from dataclasses import dataclass
from abc import abstractmethod
import numpy as np

from zeroth.core.common.optimizer import Optimizer
from zeroth.core.spsa.perturbations import Perturbation


@dataclass(frozen=True)
class OptimizerPerturbationConfig:
    learning_rate: float

    def instantiate(self, perturbation: Perturbation):
        return NotImplementedError


@dataclass(frozen=True)
class SGDPerturbationConfig(OptimizerPerturbationConfig):
    name = "SGD"
    def instantiate(self, perturbation: Perturbation):
        return SGDPerturbation(self, perturbation)


@dataclass(frozen=True)
class AdamPerturbationConfig(OptimizerPerturbationConfig):
    name = "Adam"
    beta1: float
    beta2: float
    epsilon: float

    def instantiate(self, perturbation: Perturbation):
        return AdamPerturbation(self, perturbation)


class OptimizerPerturbation(Optimizer):
    """Abstract base class for optimizers using Stochastic Perturbation (spsa).

    Instead of calculating gradients via backpropagation, these optimizers estimate
    the gradient by evaluating the loss on perturbed versions of the parameters.
    """
    def __init__(self, perturbation: Perturbation, learning_rate: float):
        self.learning_rate = learning_rate
        self.perturbation = perturbation

    def get_gradient(self, pLoss, batch_loss):
        """Estimates the gradient from loss differences.

        Args:
            pLoss (np.ndarray): Losses of perturbed models. Shape: (nb_perturbations, batch_size).
            batch_loss (np.ndarray): Loss of the nominal (unperturbed) model. Shape: (1, batch_size).

        Returns:
            np.ndarray: Estimated gradient vector. Shape: (nb_params, ).
        """
        batch_size = pLoss.shape[1]
        L_diff = pLoss - batch_loss
        return self.perturbation.get_gradient(L_diff, batch_size)

    def do_descent(self, neural_network, loss, data, batch_idx):
        """Performs one optimization step using spsa.

        1. Computes nominal prediction Y_pred.
        2. Computes perturbed predictions pY_pred (parallelized).
        3. Calculates losses for both.
        4. Estimates gradient and updates parameters.

        Returns:
            float: Average loss over the batch
        """
        X = data.X_train[batch_idx]
        Y_pred = neural_network.get_output(X)
        pY_pred = neural_network.get_p_output(X, self.perturbation)

        avg_loss, batch_loss, pLoss = loss.get_losses_for_perturbation(Y_pred, pY_pred, data, batch_idx)
        grad = self.get_gradient(pLoss, batch_loss)

        final_grad = self.apply_update_rule(grad)
        neural_network.update_params(final_grad, self.learning_rate)
        return avg_loss

    @abstractmethod
    def apply_update_rule(self, grad):
        pass


class SGDPerturbation(OptimizerPerturbation):
    def __init__(self, config: SGDPerturbationConfig, perturbation: Perturbation):
        super().__init__(perturbation, config.learning_rate)

    def apply_update_rule(self, grad):
        return grad


class AdamPerturbation(OptimizerPerturbation):
    """Adaptive Moment Estimation (Adam) adapted for spsa gradient estimates.

    Note:
        Since spsa gradients are noisy approximations, Adam is often very effective
        as its momentum terms (m, v) help smooth out the noise over time.
    """
    def __init__(self, config: AdamPerturbationConfig, perturbation: Perturbation):
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.beta1t = config.beta1
        self.beta2t = config.beta2
        self.m = 0
        self.v = 0

        super().__init__(perturbation, config.learning_rate)

    def apply_update_rule(self, grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1t)
        v_hat = self.v / (1 - self.beta2t)
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return m_hat / (np.sqrt(v_hat) + self.epsilon)
