from zeroth.core import ModelBackpropagationConfig, ModelPerturbationConfig
from zeroth.core.utils import metric_mnist
import zeroth.core.common.losses as losses
from .config import NETWORKS, OPTIMIZERS, PERTURBATIONS

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCatalog:
    backprop_linear_sgd: ModelBackpropagationConfig = ModelBackpropagationConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        optimizer_config=OPTIMIZERS.SGDBackpropagation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )

    backprop_linear_adam: ModelBackpropagationConfig = ModelBackpropagationConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        optimizer_config=OPTIMIZERS.AdamBackpropagation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )

    backprop_xs_adam: ModelBackpropagationConfig = ModelBackpropagationConfig(
        name="backprop_xs_adam",
        id={},
        neural_network_config=NETWORKS.XS,
        optimizer_config=OPTIMIZERS.AdamBackpropagation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )

    backprop_xs_sgd: ModelBackpropagationConfig = ModelBackpropagationConfig(
        name="backprop_xs_sgd",
        id={},
        neural_network_config=NETWORKS.XS,
        optimizer_config=OPTIMIZERS.SGDBackpropagation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )

    backprop_s_adam_5epochs: ModelBackpropagationConfig = ModelBackpropagationConfig(
        name="backprop_s_adam_5epochs",
        id={},
        neural_network_config=NETWORKS.S,
        optimizer_config=OPTIMIZERS.AdamBackpropagation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=5,
    )

    multiplex_linear_adam: ModelPerturbationConfig = ModelPerturbationConfig(
        name="multiplex_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        perturbation_config=PERTURBATIONS.Multiplex,
        optimizer_config=OPTIMIZERS.AdamPerturbation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )

    multiplex_linear_sgd: ModelPerturbationConfig = ModelPerturbationConfig(
        name="perturb_linear_sgd",
        id={},
        neural_network_config=NETWORKS.Linear,
        perturbation_config=PERTURBATIONS.Multiplex,
        optimizer_config=OPTIMIZERS.SGDPerturbation,
        loss=losses.CrossEntropy(),
        metric=metric_mnist,
        batch_size=50,
        nb_epochs=1,
    )


MODELS = ModelCatalog()
