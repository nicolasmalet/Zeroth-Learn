import unittest

import numpy as np

from zeroth.utils.perturbation_matrices import RademacherMatrix
from zeroth.zeroth_order.gradient_estimators import SimultaneousPerturbation


class SimultaneousPerturbationTest(unittest.TestCase):
    def test_recovers_linear_gradient(self):
        np.random.seed(42)
        expected = np.array([1.5, -2.0, 0.5])
        theta = np.array([0.2, -0.1, 0.3])
        estimator = SimultaneousPerturbation(1e-6, 2_000, RademacherMatrix(), theta.size)

        perturbed_theta = estimator.perturb(theta)
        losses = (perturbed_theta @ expected)[:, None]

        np.testing.assert_allclose(estimator.get_gradient(losses), expected, atol=0.06)


if __name__ == "__main__":
    unittest.main()
