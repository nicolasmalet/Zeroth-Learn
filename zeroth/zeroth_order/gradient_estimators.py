from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..types import Array


@dataclass(frozen=True)
class GradientEstimatorConfig:
    def instantiate(self, nb_params: int) -> GradientEstimator:
        pass


@dataclass(frozen=True)
class NullGradientEstimatorConfig(GradientEstimatorConfig):
    def instantiate(self, nb_params: int) -> NullGradientEstimator:
        return NullGradientEstimator(nb_params=nb_params)


@dataclass(frozen=True)
class FiniteDifferenceConfig(GradientEstimatorConfig):
    dA: float

    def instantiate(self, nb_params) -> FiniteDifference:
        return FiniteDifference(self, nb_params)


@dataclass(frozen=True)
class SimultaneousPerturbationConfig(GradientEstimatorConfig):
    dA: float
    nb_perturbations: int
    get_perturbation_matrix: Callable

    def instantiate(self, nb_params: int) -> SimultaneousPerturbation:
        return SimultaneousPerturbation(self, nb_params)


class GradientEstimator(ABC):
    @abstractmethod
    def perturb(self, Theta: Array) -> Array:
        """Applies the perturbation to the parameter vector Theta.

        Args:
            Theta (Array): The flat parameter vector Theta, shape: (nb_params, )

        Returns:
            Array: Perturbed parameters pThetas. Shape (T, nb_params).
                        T is the number of perturbations (batch of models).
        """
        pass

    @abstractmethod
    def get_gradient(self, p_Loss: Array) -> Array:
        """Estimates the gradient using the perturbation method.

        Args:
            p_Loss (Array): loss of the perturbed network (perturbed Loss - Loss).
                                 Shape: (T + 1, batch_size).

        Returns:
            Array: Estimated gradient vector. Shape (nb_params, ).
        """
        pass


class NullGradientEstimator(GradientEstimator):
    def __init__(self, nb_params: int) -> None:
        self.nb_params: int = nb_params

    def perturb(self, Theta: Array) -> Array:
        return Theta[None, :]

    def get_gradient(self, p_Loss: Array) -> Array:
        return np.zeros(self.nb_params)


class FiniteDifference(GradientEstimator):
    def __init__(self, config: FiniteDifferenceConfig, nb_params: int) -> None:
        self.nb_params: int = nb_params
        self.dA: float = config.dA

        self.perturbation_matrix: Array = np.vstack((np.zeros((1, self.nb_params)), np.eye(nb_params)))
        self.Ps: Array = config.dA * self.perturbation_matrix

    def perturb(self, Theta: Array) -> Array:
        return Theta + self.Ps

    def get_gradient(self, p_Loss: Array) -> Array:
        L_diff = p_Loss[1:] - p_Loss[0]
        return np.mean(L_diff, axis=1) / self.dA


class SimultaneousPerturbation(GradientEstimator):
    def __init__(self, config: SimultaneousPerturbationConfig, nb_params: int) -> None:
        self.nb_params: int = nb_params
        self.dA: float = config.dA
        self.nb_perturbations: int = config.nb_perturbations
        self.get_perturbation_matrix: Callable = config.get_perturbation_matrix

        nb_copies = 3
        self.Ps_extended: Array = np.vstack((np.zeros((1, self.nb_params * nb_copies)),
                                             self.get_perturbation_matrix(self.nb_perturbations,
                                                                          self.nb_params * nb_copies)))
        self.max_offset: int = self.Ps_extended.shape[1] - self.nb_params

        self._perturbed_params: Array = np.empty((self.nb_perturbations + 1, self.nb_params))
        self.last_offset: int = 0

    def perturb(self, Theta: Array) -> Array:
        self.last_offset = np.random.randint(0, self.max_offset)
        Ps = self.Ps_extended[:, self.last_offset:self.last_offset + self.nb_params]
        np.multiply(self.dA, Ps, out=self._perturbed_params)
        self._perturbed_params += Theta

        return self._perturbed_params

    def get_gradient(self, p_Loss: Array) -> Array:
        L_diff = p_Loss[1:] - p_Loss[0]
        Ps = self.Ps_extended[1:, self.last_offset:self.last_offset + self.nb_params]
        return Ps.T @ L_diff.mean(axis=1) / self.dA
