from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np

from .gradient_estimators import GradientEstimator
from .zeroth_order_blackbox import ZerothOrderBlackBox
from ..abstract import Loss, Optimizer, Summary
from ..types import Array


@dataclass(frozen=True)
class ZerothOrderOptimizerConfig(Summary, ABC):
    @abstractmethod
    def instantiate(self, gradient_estimator: GradientEstimator) -> ZerothOrderOptimizer:
        ...


@dataclass(frozen=True)
class ZerothOrderSGDConfig(ZerothOrderOptimizerConfig):
    name = "SGD"
    learning_rate: float

    def instantiate(self, gradient_estimator: GradientEstimator) -> ZerothOrderSGD:
        return ZerothOrderSGD(self, gradient_estimator)


@dataclass(frozen=True)
class ZerothOrderAdamConfig(ZerothOrderSGDConfig):
    name = "Adam"
    beta1: float
    beta2: float
    epsilon: float

    def instantiate(self, gradient_estimator: GradientEstimator) -> ZerothOrderAdam:
        return ZerothOrderAdam(self, gradient_estimator)


class ZerothOrderOptimizer(Optimizer):
    @abstractmethod
    def do_descent(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: Array, Y_true: Array) -> float:
        ...

    @abstractmethod
    def compute_gradient(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: Array, Y_true: Array) -> tuple[
        float, Array]:
        ...

    @abstractmethod
    def update_params(self, blackbox: ZerothOrderBlackBox, gradient: Array) -> None:
        ...


class ZerothOrderSGD(ZerothOrderOptimizer):
    """Abstract base class for optimizers using Stochastic Perturbation (zeroth_order).

    Instead of calculating gradients via first_order, these optimizers estimate
    the gradient by evaluating the loss on perturbed versions of the parameters.
    """

    def __init__(self, config: ZerothOrderSGDConfig, gradient_estimator: GradientEstimator) -> None:
        self.learning_rate = config.learning_rate
        self.gradient_estimator = gradient_estimator

    def do_descent(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: Array, Y_true: Array) -> float:
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

    def compute_gradient(self, blackbox: ZerothOrderBlackBox, loss: Loss, X: Array, Y_true: Array) -> tuple[
        float, Array]:
        pY_pred = blackbox.forward_perturbed(X, self.gradient_estimator)
        avg_loss, pLoss = loss.compute_losses_for_zeroth_order(pY_pred, Y_true)
        gradient = self.gradient_estimator.get_gradient(pLoss)
        return avg_loss, gradient

    def update_params(self, blackbox: ZerothOrderBlackBox, gradient: Array) -> None:
        final_gradient = self._apply_update_rule(gradient)
        blackbox.update_params(final_gradient, self.learning_rate)

    def _apply_update_rule(self, grad: Array) -> Array:
        return grad


class ZerothOrderAdam(ZerothOrderSGD):
    name = "Adam"
    """Adaptive Moment Estimation (Adam) adapted for zeroth_order gradient estimates.

    Note:
        Since zeroth_order gradients are noisy approximations, Adam is often very effective
        as its momentum terms (m, v) help smooth out the noise over time.
    """

    def __init__(self, config: ZerothOrderAdamConfig, gradient_estimator: GradientEstimator) -> None:
        self.beta1: float = config.beta1
        self.beta2: float = config.beta2
        self.epsilon: float = config.epsilon
        self.beta1t: float = config.beta1
        self.beta2t: float = config.beta2
        self.m: Array = np.array([0])
        self.v: Array = np.array([0])

        super().__init__(config, gradient_estimator)

    def _apply_update_rule(self, grad: Array) -> Array:
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m / (1 - self.beta1t)
        v_hat = self.v / (1 - self.beta2t)
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return m_hat / (np.sqrt(v_hat) + self.epsilon)
