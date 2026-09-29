from __future__ import annotations

from dataclasses import dataclass

from ..abstract import Model, ModelConfig, NeuralNetworkConfig
from ..data import Data
from .neural_network import FirstOrderNeuralNetwork
from .optimizers import FirstOrderOptimizerConfig


@dataclass(frozen=True, kw_only=True)
class FirstOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: FirstOrderOptimizerConfig

    def instantiate(self, data: Data) -> FirstOrderModel:
        return FirstOrderModel(self, data)


class FirstOrderModel(Model):
    def __init__(self, config: FirstOrderModelConfig, data: Data) -> None:
        super().__init__(config, data)

        self.neural_network = FirstOrderNeuralNetwork(config.neural_network_config, data.input_dim, data.output_dim)
        self.optimizer = config.optimizer_config.instantiate()
