from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for all optimization algorithms.
    """

    @abstractmethod
    def do_descent(self, neural_network, loss, data, batch_index):
        """Performs a single optimization step.

        This includes:
        1. Forward pass (prediction)
        2. Gradient calculation (analytical or estimated)
        3. Parameter update

        Args:
            neural_network (NeuralNetwork): The model to train.
            loss (Loss): The loss function object.
            data (Data): The dataset handler.
            batch_index (int): The index of the current batch.

        Returns:
            float: The average loss for the processed batch.
        """
        raise NotImplementedError
