import numpy as np


def rademacher_matrix(nb_perturbation: int, nb_parameters: int) -> np.ndarray:
    S = np.random.choice([-1, 1], size=(nb_perturbation, nb_parameters))
    P = S / np.sqrt(nb_perturbation)  # columns have exact L2 norm = 1
    return P
