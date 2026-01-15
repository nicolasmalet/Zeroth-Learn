from dataclasses import dataclass
from typing import Callable
from abc import ABC
import numpy as np


@dataclass(frozen=True)
class PerturbationConfig:
    dA: float

    def instantiate(self, nb_params):
        return NotImplementedError


@dataclass(frozen=True)
class OneByOneConfig(PerturbationConfig):
    def instantiate(self, nb_params):
        return OneByOne(self, nb_params)


@dataclass(frozen=True)
class MultiplexConfig(PerturbationConfig):
    nb_perturbations: int
    get_perturbation_matrix: Callable

    def instantiate(self, nb_params):
        return Multiplex(self, nb_params)


class Perturbation(ABC):
    def __init__(self, nb_params, dA):

        self.nb_params = nb_params
        self.dA = dA
        self.Ps = None

    def perturb(self, params):
        """Applies the perturbation to the parameter vector Theta.

        Args:
            params (Params): The object holding the flat parameter vector Theta.
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


class OneByOne(Perturbation):
    def __init__(self, config: OneByOneConfig, nb_params):
        super().__init__(nb_params, config.dA)
        self.nb_params = nb_params

        # each parameter is perturbed one after another :
        self.perturbation_matrix = np.eye(nb_params)
        self.Ps = config.dA * self.perturbation_matrix

    def get_gradient(self, L_diff, batch_size):
        return np.mean(L_diff, axis=1) / self.dA


class Multiplex(Perturbation):
    def __init__(self, config: MultiplexConfig, nb_params):
        super().__init__(nb_params, config.dA)
        self.nb_perturbations = config.nb_perturbations
        self.get_perturbation_matrix = config.get_perturbation_matrix

    def perturb(self, params):
        # we change the perturbation matrix at each iteration :
        self.Ps = self.dA * self.get_perturbation_matrix(self.nb_perturbations, self.nb_params)
        return params.Theta + self.Ps
