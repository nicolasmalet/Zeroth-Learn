import numpy as np

from zeroth.abstract.blackbox import BlackBox
from zeroth.zeroth_order.gradient_estimators import GradientEstimator
from abc import abstractmethod

class ZerothOrderBlackBox(BlackBox):

    @abstractmethod
    def forward_perturbed(self, X: np.ndarray, gradient_estimator: GradientEstimator) -> np.ndarray:
        pass

    @abstractmethod
    def update_params(self, grad: np.ndarray, learning_rate: float) -> None:
        pass