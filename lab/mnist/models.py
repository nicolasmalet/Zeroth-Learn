from zeroth import FirstOrderModelConfig, ZerothOrderModelConfig
from zeroth.utils.metrics import accuracy
import zeroth.losses as losses

from .configs import NETWORKS, OPTIMIZERS, GRADIENT_ESTIMATORS

from dataclasses import dataclass

DEFAULT_BATCH_SIZE = 50
DEFAULT_NB_EPOCHS = 1


@dataclass(frozen=True)
class ModelCatalog:
    backprop_linear_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.linear,
        optimizer_config=OPTIMIZERS.first_order_sgd,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_linear_adam: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_linear_adam",
        id={},
        neural_network_config=NETWORKS.linear,
        optimizer_config=OPTIMIZERS.first_order_adam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_xs_adam: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_xs_adam",
        id={},
        neural_network_config=NETWORKS.xs,
        optimizer_config=OPTIMIZERS.first_order_adam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_xs_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_xs_sgd",
        id={},
        neural_network_config=NETWORKS.xs,
        optimizer_config=OPTIMIZERS.first_order_sgd,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    backprop_s_adam_5epochs: FirstOrderModelConfig = FirstOrderModelConfig(
        name="backprop_s_adam_5epochs",
        id={},
        neural_network_config=NETWORKS.s,
        optimizer_config=OPTIMIZERS.first_order_adam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    multiplex_linear_adam: ZerothOrderModelConfig = ZerothOrderModelConfig(
        name="multiplex_linear_adam",
        id={},
        neural_network_config=NETWORKS.linear,
        gradient_estimator_config=GRADIENT_ESTIMATORS.simultaneous_perturbation,
        optimizer_config=OPTIMIZERS.zeroth_order_adam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    multiplex_linear_sgd: ZerothOrderModelConfig = ZerothOrderModelConfig(
        name="perturb_linear_sgd",
        id={},
        neural_network_config=NETWORKS.linear,
        gradient_estimator_config=GRADIENT_ESTIMATORS.simultaneous_perturbation,
        optimizer_config=OPTIMIZERS.zeroth_order_sgd,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )

    finite_difference_linear_adam: ZerothOrderModelConfig = ZerothOrderModelConfig(
        name="multiplex_linear_adam",
        id={},
        neural_network_config=NETWORKS.linear,
        gradient_estimator_config=GRADIENT_ESTIMATORS.finite_difference,
        optimizer_config=OPTIMIZERS.zeroth_order_adam,
        loss=losses.CrossEntropy(),
        metric=accuracy,
        batch_size=DEFAULT_BATCH_SIZE,
        nb_epochs=DEFAULT_NB_EPOCHS,
    )


MODELS = ModelCatalog()
