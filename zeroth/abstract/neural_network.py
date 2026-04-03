from abc import ABC
from dataclasses import dataclass

from .activation import Activation


@dataclass(frozen=True)
class LayerConfig:
    input_dim: int
    output_dim: int
    activation: Activation


@dataclass(frozen=True)
class NeuralNetworkConfig:
    name: str
    layers_config: list[LayerConfig]


class NeuralNetwork(ABC):
    def __init__(self, config: NeuralNetworkConfig):
        self.name: str = config.name
        self.nb_layers: int = len(config.layers_config)
        self.input_dim: int = config.layers_config[0].input_dim
        self.output_dim: int = config.layers_config[-1].output_dim
