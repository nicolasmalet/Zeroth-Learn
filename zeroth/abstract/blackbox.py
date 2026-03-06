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


class BlackBox(ABC):
    """
    Abstract Base Class defining the required interface for any Neural Network implementation.

    Whether the network uses Backpropagation or zeroth_order (Perturbation), it must implement
    these methods to be compatible with the Model and Experiment classes.
    """

    @abstractmethod
    def init_params(self, params: tuple):
        """Manually initializes the weights and biases of the network.

        Args:
            params (tuple): Tuple containing the parameters (eg: weights and biases).
        """
        pass

    @abstractmethod
    def get_params(self) -> tuple:
        """Retrieves the current parameters of the network.

        Returns:
            Tuple: (eg: List of Weight matrices, List of Bias vectors).
        """
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Computes the forward pass.

        Args:
            X (np.ndarray): Input data. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Network predictions. Shape: (output_dim, batch_size).
        """
        pass
