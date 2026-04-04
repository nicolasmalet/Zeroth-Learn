from abc import abstractmethod

from .gradient_estimators import GradientEstimator
from ..abstract import BlackBox
from ..types import Array


class ZerothOrderBlackBox(BlackBox):

    @abstractmethod
    def forward_perturbed(self, X: Array, gradient_estimator: GradientEstimator) -> Array:
        ...

    @abstractmethod
    def update_params(self, grad: Array, learning_rate: float) -> None:
        ...
