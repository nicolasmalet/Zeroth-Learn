from __future__ import annotations

from dataclasses import dataclass

from .gradient_estimators import GradientEstimatorConfig, GradientEstimator
from .neural_network.neural_network import ZerothOrderNeuralNetwork
from .optimizers import ZerothOrderOptimizerConfig, ZerothOrderOptimizer
from ..abstract import Model, ModelConfig, NeuralNetworkConfig
from ..data import Data


@dataclass(frozen=True)
class ZerothOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: ZerothOrderOptimizerConfig
    gradient_estimator_config: GradientEstimatorConfig

    def instantiate(self, data: Data) -> ZerothOrderModel:
        return ZerothOrderModel(self, data)


class ZerothOrderModel(Model):
    def __init__(self, config: ZerothOrderModelConfig, data: Data) -> None:
        super().__init__(config, data)

        self.neural_network: ZerothOrderNeuralNetwork = ZerothOrderNeuralNetwork(config.neural_network_config, data.input_dim, data.output_dim)
        nb_params = self.neural_network.params.nb_params
        self.gradient_estimator: GradientEstimator = config.gradient_estimator_config.instantiate(nb_params)
        self.optimizer: ZerothOrderOptimizer = config.optimizer_config.instantiate(self.gradient_estimator)
