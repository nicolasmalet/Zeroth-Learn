from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    """

    @abstractmethod
    def do_descent(self, neural_network, loss, X: np.ndarray, Y_true: np.ndarray) -> float:
        """Performs a single optimization step.

        This includes:
        1. Forward pass (prediction)
        2. Gradient calculation (analytical or estimated)
        3. Parameter update

        Args:
            neural_network (NeuralNetwork): The model to train.
            loss (Loss): The loss function object.
            X (np.ndarray): The input data.
            Y_true (int): The true label.

        Returns:
            float: The average loss for the processed batch.
        """
        pass
