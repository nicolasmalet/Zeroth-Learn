from dataclasses import dataclass
from typing import Callable
from abc import ABC
import numpy as np


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

    def perturb(self, Theta: np.ndarray):
        """Applies the perturbation to the parameter vector Theta.

        Args:
            Theta (np.ndarray): The flat parameter vector Theta, shape: (nb_params, )

        Returns:
            np.ndarray: Perturbed parameters pThetas. Shape (T, nb_params).
                        T is the number of perturbations (batch of models).
        """
        return Theta + self.Ps

    def get_gradient(self, p_Loss: np.ndarray) -> np.ndarray:
        """Estimates the gradient using the perturbation method.

        Args:
            p_Loss (np.ndarray): loss of the perturbed network (perturbed Loss - Loss).
                                 Shape: (T + 1, batch_size).

        Returns:
            np.ndarray: Estimated gradient vector. Shape (nb_params, ).
        """

        L_diff = p_Loss[1:] - p_Loss[0]
        return self.Ps.T @ L_diff.mean(axis=1) / (self.dA**2)


class FiniteDifference(GradientEstimator):
    def __init__(self, config: FiniteDifferenceConfig, nb_params: int):
        super().__init__(nb_params, config.dA)
        self.nb_params: int = nb_params

        # each parameter is perturbed one after another :
        self.perturbation_matrix: np.ndarray = np.vstack((np.zeros((1, self.nb_params)), np.eye(nb_params)))
        self.Ps: np.ndarray = config.dA * self.perturbation_matrix

    def get_gradient(self, p_Loss: np.ndarray):
        L_diff = p_Loss[1:] - p_Loss[0]
        return np.mean(L_diff, axis=1) / self.dA


class SimultaneousPerturbation(GradientEstimator):
    def __init__(self, config: SimultaneousPerturbationConfig, nb_params: int):
        super().__init__(nb_params, config.dA)
        self.nb_perturbations: int = config.nb_perturbations
        self.get_perturbation_matrix: Callable = config.get_perturbation_matrix

        nb_copies = 3
        self.Ps_extended = np.vstack((np.zeros((1, self.nb_params * nb_copies)), self.get_perturbation_matrix(self.nb_perturbations, self.nb_params * nb_copies)))
        self.max_offset = self.Ps_extended.shape[1] - self.nb_params

        self._perturbed_params = np.empty((self.nb_perturbations + 1, self.nb_params))
        self.last_offset = 0

    def perturb(self, Theta: np.ndarray) -> np.ndarray:
        self.last_offset = np.random.randint(0, self.max_offset)
        Ps = self.Ps_extended[:, self.last_offset:self.last_offset + self.nb_params]
        np.multiply(self.dA, Ps, out=self._perturbed_params)
        self._perturbed_params += Theta

        return self._perturbed_params

    def get_gradient(self, p_Loss: np.ndarray) -> np.ndarray:
        L_diff = p_Loss[1:] - p_Loss[0]
        Ps = self.Ps_extended[1:, self.last_offset:self.last_offset + self.nb_params]
        return Ps.T @ L_diff.mean(axis=1) / self.dA