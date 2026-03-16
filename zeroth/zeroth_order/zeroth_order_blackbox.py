from abc import abstractmethod

import numpy as np

from .gradient_estimators import GradientEstimator
from ..abstract.blackbox import BlackBox


class ZerothOrderBlackBox(BlackBox):

    @abstractmethod
    def forward_perturbed(self, X: np.ndarray, gradient_estimator: GradientEstimator) -> np.ndarray:
        pass

    @abstractmethod
    def update_params(self, grad: np.ndarray, learning_rate: float) -> None:
        pass