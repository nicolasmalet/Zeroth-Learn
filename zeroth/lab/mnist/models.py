from zeroth.core import FirstOrderModelConfig, ZerothOrderModelConfig
from zeroth.core.utils.metrics import accuracy
import zeroth.core.losses as losses

from .config import NETWORKS, OPTIMIZERS, GRADIENT_ESTIMATORS

from dataclasses import dataclass

DEFAULT_BATCH_SIZE = 50
DEFAULT_NB_EPOCHS = 1


@dataclass(frozen=True)
class ModelCatalog:
    backprop_linear_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        optimizer_config=OPTIMIZERS.FirstOrderSGD,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_linear_adam: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        optimizer_config=OPTIMIZERS.FirstOrderAdam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_xs_adam: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_xs_adam",
        id={},
        neural_network_config=NETWORKS.XS,
        optimizer_config=OPTIMIZERS.FirstOrderAdam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_xs_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_xs_sgd",
        id={},
        neural_network_config=NETWORKS.XS,
        optimizer_config=OPTIMIZERS.FirstOrderSGD,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_s_adam_5epochs: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_s_adam_5epochs",
        id={},
        neural_network_config=NETWORKS.S,
        optimizer_config=OPTIMIZERS.FirstOrderAdam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    multiplex_linear_adam: ZerothOrderModelConfig = ZerothOrderModelConfig(
        name="multiplex_linear_adam",
        id={},
        neural_network_config=NETWORKS.Linear,
        gradient_estimator_config=GRADIENT_ESTIMATORS.SimultaneousPerturbation,
        optimizer_config=OPTIMIZERS.ZerothOrderAdam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    multiplex_linear_sgd: ZerothOrderModelConfig = ZerothOrderModelConfig(
        name="perturb_linear_sgd",
        id={},
        neural_network_config=NETWORKS.Linear,
        gradient_estimator_config=GRADIENT_ESTIMATORS.SimultaneousPerturbation,
        optimizer_config=OPTIMIZERS.ZerothOrderSGD,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )


MODELS = ModelCatalog()
