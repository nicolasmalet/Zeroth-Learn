from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class LayerConfig:
    input_dim: int
    output_dim: int
    f: Callable


@dataclass(frozen=True)
class NeuralNetworkConfig:
    name: str
    layers_config: list[LayerConfig]


class NeuralNetwork(ABC):
    """
    Abstract Base Class defining the required interface for any Neural Network implementation.

    Whether the network uses Backpropagation or spsa (Perturbation), it must implement
    these methods to be compatible with the Model and Experiment classes.
    """

    @abstractmethod
    def init_params(self, Ws: list[np.ndarray], Bs: list[np.ndarray]):
        """Manually initializes the weights and biases of the network.

        Args:
            Ws (List[np.ndarray]): List of weight matrices for each layer.
            Bs (List[np.ndarray]): List of bias vectors for each layer.
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Retrieves the current parameters of the network.

        Returns:
            Tuple: (List of Weight matrices, List of Bias vectors).
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self, X: np.ndarray) -> np.ndarray:
        """Computes the forward pass.

        Args:
            X (np.ndarray): Input data. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Network predictions. Shape: (output_dim, batch_size).
        """
        raise NotImplementedError
