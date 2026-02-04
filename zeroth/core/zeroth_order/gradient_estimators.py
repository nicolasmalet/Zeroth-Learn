from dataclasses import dataclass
from typing import Callable
from abc import ABC
import numpy as np

from zeroth.core.zeroth_order.parameter_manager import ParameterManager


@dataclass(frozen=True)
class GradientEstimatorConfig:
    dA: float

    def instantiate(self, nb_params: int):
        return NotImplementedError


@dataclass(frozen=True)
class FiniteDifferenceConfig(GradientEstimatorConfig):
    def instantiate(self, nb_params):
        return FiniteDifference(self, nb_params)


@dataclass(frozen=True)
class SimultaneousPerturbationConfig(GradientEstimatorConfig):
    nb_perturbations: int
    get_perturbation_matrix: Callable

    def instantiate(self, nb_params: int):
        return SimultaneousPerturbation(self, nb_params)


class GradientEstimator(ABC):
    def __init__(self, nb_params, dA):

        self.nb_params: int = nb_params
        self.dA: float = dA
        self.Ps: np.ndarray = np.ndarray([])

    def perturb(self, params):
        """Applies the perturbation to the parameter vector Theta.

        Args:
            params (ParameterManager): The object holding the flat parameter vector Theta.
                             Theta shape: (nb_params, )

        Returns:
            np.ndarray: Perturbed parameters pThetas. Shape (T, nb_params).
                        T is the number of perturbations (batch of models).
        """
        return params.Theta + self.Ps

    def get_gradient(self, L_diff: np.ndarray, batch_size: int) -> np.ndarray:
        """Estimates the gradient using the perturbation method.

        Args:
            L_diff (np.ndarray): Difference in loss (perturbed Loss - Loss).
                                 Shape: (T, batch_size).
            batch_size (int): Size of the data batch used.

        Returns:
            np.ndarray: Estimated gradient vector. Shape (nb_params, ).
        """
        # Einstein summation to compute the gradient estimation efficiently
        # ij -> L_diff (T, batch), ik -> Ps (T, params)
        Sum = np.einsum('ij,ik->ik', L_diff, self.Ps) / batch_size
        return np.sum(Sum, axis=0) / self.dA ** 2


class FiniteDifference(GradientEstimator):
    def __init__(self, config: FiniteDifferenceConfig, nb_params: int):
        super().__init__(nb_params, config.dA)
        self.nb_params: int = nb_params

        # each parameter is perturbed one after another :
        self.perturbation_matrix: np.ndarray = np.eye(nb_params)
        self.Ps: np.ndarray = config.dA * self.perturbation_matrix

    def get_gradient(self, L_diff: np.ndarray, batch_size: int):
        return np.mean(L_diff, axis=1) / self.dA


class SimultaneousPerturbation(GradientEstimator):
    def __init__(self, config: SimultaneousPerturbationConfig, nb_params: int):
        super().__init__(nb_params, config.dA)
        self.nb_perturbations: int = config.nb_perturbations
        self.get_perturbation_matrix: Callable = config.get_perturbation_matrix

    def perturb(self, params: ParameterManager):
        # we change the perturbation matrix at each iteration :
        self.Ps = self.dA * self.get_perturbation_matrix(self.nb_perturbations, self.nb_params)
        return params.Theta + self.Ps
