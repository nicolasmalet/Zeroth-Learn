from dataclasses import dataclass

from .neural_network import FirstOrderNeuralNetwork
from .optimizers import FirstOrderOptimizerConfig, FirstOrderOptimizer
from ..abstract import ModelConfig, Model, NeuralNetworkConfig


@dataclass(frozen=True)
class FirstOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: FirstOrderOptimizerConfig

    def instantiate(self):
        return FirstOrderModel(self)


class FirstOrderModel(Model):
    def __init__(self, config: FirstOrderModelConfig):
        super().__init__(config)

        self.neural_network: FirstOrderNeuralNetwork = FirstOrderNeuralNetwork(config.neural_network_config)
        self.optimizer: FirstOrderOptimizer = config.optimizer_config.instantiate()
