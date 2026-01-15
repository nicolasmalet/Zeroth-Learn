from dataclasses import dataclass

import zeroth.core.activation_functions as af
from zeroth.core import backpropagation as bp, spsa
from zeroth.core.utils import rademacher_matrix
from zeroth.core.common import LayerConfig, NeuralNetworkConfig


INPUT_DIM = 28 ** 2

@dataclass(frozen=True)
class PerturbationCatalog:
    OneByOne: spsa.OneByOneConfig = spsa.OneByOneConfig(dA=1e-8)
    Multiplex: spsa.MultiplexConfig = spsa.MultiplexConfig(dA=1e-8,
                                                           nb_perturbations=50,
                                                           get_perturbation_matrix=rademacher_matrix)


PERTURBATIONS = PerturbationCatalog()


@dataclass(frozen=True)
class OptimizerCatalog:
    SGDBackpropagation: bp.SGDBackpropagationConfig = bp.SGDBackpropagationConfig(learning_rate=0.1)
    AdamBackpropagation: bp.AdamBackpropagationConfig = bp.AdamBackpropagationConfig(learning_rate=0.001,
                                                       beta1=0.9,
                                                       beta2=0.99,
                                                       epsilon=1e-8)
    SGDPerturbation: spsa.SGDPerturbationConfig = spsa.SGDPerturbationConfig(learning_rate=0.1)
    AdamPerturbation: spsa.AdamPerturbationConfig = spsa.AdamPerturbationConfig(learning_rate=0.001,
                                                   beta1=0.9,
                                                   beta2=0.99,
                                                   epsilon=1e-8)


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
