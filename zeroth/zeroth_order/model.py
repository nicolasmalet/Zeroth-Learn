from ..abstract import Model, ModelConfig, NeuralNetworkConfig
from .gradient_estimators import GradientEstimatorConfig, GradientEstimator
from .optimizers import ZerothOrderOptimizerConfig, ZerothOrderOptimizer
from .neural_network.neural_network import ZerothOrderNeuralNetwork

from dataclasses import dataclass


@dataclass(frozen=True)
class ZerothOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: ZerothOrderOptimizerConfig
    gradient_estimator_config: GradientEstimatorConfig

    def instantiate(self):
        return ZerothOrderModel(self)


class ZerothOrderModel(Model):
    def __init__(self, config: ZerothOrderModelConfig):
        super().__init__(config)

        self.neural_network: ZerothOrderNeuralNetwork = ZerothOrderNeuralNetwork(config.neural_network_config)
        nb_params = self.neural_network.params.nb_params
        self.gradient_estimator: GradientEstimator = config.gradient_estimator_config.instantiate(nb_params)
        self.optimizer: ZerothOrderOptimizer = config.optimizer_config.instantiate(self.gradient_estimator)
