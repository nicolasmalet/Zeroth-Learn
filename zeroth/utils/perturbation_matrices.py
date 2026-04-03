from typing import Self

import numpy as np

from ..abstract.perturbation_matrix import PerturbationMatrix
from ..types import Array


class RademacherMatrix(PerturbationMatrix):
    def __call__(self: Self, nb_perturbation: int, nb_parameters: int) -> Array:
        S = 2 * np.random.randint(0, 2, size=(nb_perturbation, nb_parameters)) - 1
        P = S / np.sqrt(nb_perturbation)  # columns have exact L2 norm = 1
        return P
