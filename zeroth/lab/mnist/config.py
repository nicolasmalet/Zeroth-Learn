from dataclasses import dataclass

import zeroth.core.utils.activation_functions as af
from zeroth.core import first_order as first, zeroth_order as zeroth
from zeroth.core.utils.perturbation_matrices import rademacher_matrix
from zeroth.core.abstract import LayerConfig, NeuralNetworkConfig

INPUT_DIM = 28 ** 2

DEFAULT_PERTURBATION_SCALE = 1e-8
DEFAULT_ADAM_LEARNING_RATE = 1e-3
DEFAULT_SGD_LEARNING_RATE = 0.1
DEFAULT_NB_PERTURBATION = 50

BETA1, BETA2 = 0.9, 0.99
EPSILON = 1e-8


@dataclass(frozen=True)
class GradientEstimatorCatalog:
    FiniteDifference: zeroth.FiniteDifferenceConfig = zeroth.FiniteDifferenceConfig(dA=DEFAULT_PERTURBATION_SCALE)
    SimultaneousPerturbation: zeroth.SimultaneousPerturbationConfig = zeroth.SimultaneousPerturbationConfig(
        dA=DEFAULT_PERTURBATION_SCALE,
        nb_perturbations=DEFAULT_NB_PERTURBATION,
        get_perturbation_matrix=rademacher_matrix)


GRADIENT_ESTIMATORS = GradientEstimatorCatalog()


@dataclass(frozen=True)
class OptimizerCatalog:
    FirstOrderSGD: first.FirstOrderSGDConfig = first.FirstOrderSGDConfig(learning_rate=DEFAULT_SGD_LEARNING_RATE)
    FirstOrderAdam: first.FirstOrderAdamConfig = first.FirstOrderAdamConfig(learning_rate=DEFAULT_ADAM_LEARNING_RATE,
                                                                            beta1=BETA1,
                                                                            beta2=BETA2,
                                                                            epsilon=EPSILON)
    ZerothOrderSGD: zeroth.ZerothOrderSGDConfig = zeroth.ZerothOrderSGDConfig(learning_rate=DEFAULT_SGD_LEARNING_RATE)
    ZerothOrderAdam: zeroth.ZerothOrderAdamConfig = zeroth.ZerothOrderAdamConfig(
        learning_rate=DEFAULT_ADAM_LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON)


OPTIMIZERS = OptimizerCatalog()


@dataclass(frozen=True)
class NetworkCatalog:
    Linear: NeuralNetworkConfig = NeuralNetworkConfig(
        name="Linear",
        layers_config=[LayerConfig(input_dim=INPUT_DIM, output_dim=10, f=af.softmax)]
    )

    XS: NeuralNetworkConfig = NeuralNetworkConfig(
        name="XS",
        layers_config=[
            LayerConfig(input_dim=INPUT_DIM, output_dim=128, f=af.relu),
            LayerConfig(input_dim=128, output_dim=10, f=af.softmax)
        ]
    )

    S: NeuralNetworkConfig = NeuralNetworkConfig(
        name="S",
        layers_config=[
            LayerConfig(input_dim=INPUT_DIM, output_dim=128, f=af.relu),
            LayerConfig(input_dim=128, output_dim=64, f=af.relu),
            LayerConfig(input_dim=64, output_dim=10, f=af.softmax)
        ]
    )

    M: NeuralNetworkConfig = NeuralNetworkConfig(
        name="M",
        layers_config=[
            LayerConfig(input_dim=INPUT_DIM, output_dim=256, f=af.relu),
            LayerConfig(input_dim=256, output_dim=128, f=af.relu),
            LayerConfig(input_dim=128, output_dim=64, f=af.relu),
            LayerConfig(input_dim=64, output_dim=10, f=af.softmax)
        ]
    )

    L: NeuralNetworkConfig = NeuralNetworkConfig(
        name="L",
        layers_config=[
            LayerConfig(input_dim=INPUT_DIM, output_dim=256, f=af.relu),
            LayerConfig(input_dim=256, output_dim=256, f=af.relu),
            LayerConfig(input_dim=256, output_dim=128, f=af.relu),
            LayerConfig(input_dim=128, output_dim=10, f=af.softmax)
        ]
    )
    XL: NeuralNetworkConfig = NeuralNetworkConfig(
        name="XL",
        layers_config=[
            LayerConfig(input_dim=INPUT_DIM, output_dim=512, f=af.relu),
            LayerConfig(input_dim=512, output_dim=512, f=af.relu),
            LayerConfig(input_dim=512, output_dim=256, f=af.relu),
            LayerConfig(input_dim=256, output_dim=128, f=af.relu),
            LayerConfig(input_dim=128, output_dim=10, f=af.softmax)
        ]
    )


NETWORKS = NetworkCatalog()
