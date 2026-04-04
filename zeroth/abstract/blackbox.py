from abc import ABC, abstractmethod

from ..types import Array


class BlackBox(ABC):
    """
    Abstract Base Class defining the required interface for any Neural Network implementation.

    Whether the network uses Backpropagation or zeroth_order (Perturbation), it must implement
    these methods to be compatible with the Model and Experiment classes.
    """

    @abstractmethod
    def __call__(self, X: Array) -> Array:
        """Computes the forward pass.

        Args:
            X (Array): Input data. Shape: (input_dim, batch_size).

        Returns:
            Array: Network predictions. Shape: (output_dim, batch_size).
        """
        ...

    @abstractmethod
    def init_params(self, params: dict) -> None:
        """Manually initializes the weights and biases of the network.

        Args:
            params (tuple): Tuple containing the parameters (eg: weights and biases).
        """
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Retrieves the current parameters of the network.

        Returns:
            dict: (eg: List of Weight matrices, List of Bias vectors).
        """
        ...
