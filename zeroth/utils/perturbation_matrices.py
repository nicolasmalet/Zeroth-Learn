import numpy as np


def rademacher_matrix(nb_perturbation: int, nb_parameters: int) -> np.ndarray:
    S = 2 * np.random.randint(0, 2, size=(nb_perturbation, nb_parameters)) - 1
    P = S / np.sqrt(nb_perturbation)  # columns have exact L2 norm = 1
    return P
