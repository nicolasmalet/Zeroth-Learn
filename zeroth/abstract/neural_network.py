from abc import ABC
from dataclasses import dataclass

from .activation import Activation
from .summary import Summary


@dataclass(frozen=True)
class NeuralNetworkConfig(Summary):
    name: str
    hidden_dims: list[int]
    activations: list[Activation]


class NeuralNetwork(ABC):
    def __init__(self, config: NeuralNetworkConfig, input_dim: int, output_dim: int):
        self.name: str = config.name
        self.nb_layers: int = len(config.activations)
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
