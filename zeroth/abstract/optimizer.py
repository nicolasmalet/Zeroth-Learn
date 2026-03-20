from abc import ABC, abstractmethod

from ..types import Array


class Optimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    """

    @abstractmethod
    def do_descent(self, neural_network, loss, X: Array, Y_true: Array) -> float:
        """Performs a single optimization step.

        This includes:
        1. Forward pass (prediction)
        2. Gradient calculation (analytical or estimated)
        3. Parameter update

        Args:
            neural_network (NeuralNetwork): The model to train.
            loss (Loss): The loss function object.
            X (Array): The input data.
            Y_true (int): The true label.

        Returns:
            float: The average loss for the processed batch.
        """
        pass
