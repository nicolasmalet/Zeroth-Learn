from dataclasses import dataclass
from abc import abstractmethod
import numpy as np

from .zeroth_order_blackbox import ZerothOrderBlackBox
from .gradient_estimators import GradientEstimator
from ..abstract.optimizer import Optimizer
from ..losses import Loss


@dataclass(frozen=True)
class ZerothOrderOptimizerConfig:
    learning_rate: float

    def instantiate(self, gradient_estimator: GradientEstimator):
        pass


@dataclass(frozen=True)
class ZerothOrderSGDConfig(ZerothOrderOptimizerConfig):
    name = "SGD"
    def instantiate(self, gradient_estimator: GradientEstimator):
        return ZerothOrderSGD(self, gradient_estimator)


@dataclass(frozen=True)
class ZerothOrderAdamConfig(ZerothOrderOptimizerConfig):
    name = "Adam"
    beta1: float
    beta2: float
    epsilon: float

    def instantiate(self, gradient_estimator: GradientEstimator):
        return ZerothOrderAdam(self, gradient_estimator)


class ZerothOrderOptimizer(Optimizer):
    """Abstract base class for optimizers using Stochastic Perturbation (zeroth_order).

    Instead of calculating gradients via first_order, these optimizers estimate
    the gradient by evaluating the loss on perturbed versions of the parameters.
    """
    def __init__(self, gradient_estimator: GradientEstimator, learning_rate: float):
        self.learning_rate = learning_rate
        self.gradient_estimator = gradient_estimator

    def do_descent(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: np.ndarray, Y_true: np.ndarray):
        """Performs one optimization step using zeroth_order.

        1. Computes nominal prediction Y_pred.
        2. Computes perturbed predictions pY_pred (parallelized).
        3. Calculates losses for both.
        4. Estimates gradient and updates parameters.

        Returns:
            float: Average loss over the batch
        """

        avg_loss, gradient = self.compute_gradient(blackbox, loss, X, Y_true)
        self.update_params(blackbox, gradient)

        return avg_loss

    def compute_gradient(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: np.ndarray, Y_true: np.ndarray) -> tuple[float, np.ndarray]:
        pY_pred = blackbox.forward_perturbed(X, self.gradient_estimator)
        avg_loss, pLoss = loss.compute_losses_for_zeroth_order(pY_pred, Y_true)
        gradient = self.gradient_estimator.get_gradient(pLoss)
        return avg_loss, gradient

    def update_params(self, blackbox: ZerothOrderBlackBox, gradient: np.ndarray) -> None:
        final_gradient = self.apply_update_rule(gradient)
        blackbox.update_params(final_gradient, self.learning_rate)

    @abstractmethod
    def apply_update_rule(self, grad: np.ndarray) -> np.ndarray:
        pass


class ZerothOrderSGD(ZerothOrderOptimizer):
    def __init__(self, config: ZerothOrderSGDConfig, gradient_estimator: GradientEstimator):
        super().__init__(gradient_estimator, config.learning_rate)

    def apply_update_rule(self, grad: np.ndarray) -> np.ndarray:
        return grad


class ZerothOrderAdam(ZerothOrderOptimizer):
    """Adaptive Moment Estimation (Adam) adapted for zeroth_order gradient estimates.

    Note:
        Since zeroth_order gradients are noisy approximations, Adam is often very effective
        as its momentum terms (m, v) help smooth out the noise over time.
    """
    def __init__(self, config: ZerothOrderAdamConfig, gradient_estimator: GradientEstimator):
        self.beta1: float = config.beta1
        self.beta2: float = config.beta2
        self.epsilon: float = config.epsilon
        self.beta1t: float = config.beta1
        self.beta2t: float = config.beta2
        self.m: np.ndarray = np.array([0])
        self.v: np.ndarray = np.array([0])

        super().__init__(gradient_estimator, config.learning_rate)

    def apply_update_rule(self, grad: np.ndarray) -> np.ndarray:
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1t)
        v_hat = self.v / (1 - self.beta2t)
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return m_hat / (np.sqrt(v_hat) + self.epsilon)
