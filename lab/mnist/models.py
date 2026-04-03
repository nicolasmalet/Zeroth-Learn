import zeroth.losses as losses
from zeroth.first_order import FirstOrderModelConfig
from zeroth.utils.metrics import Accuracy
from zeroth.zeroth_order import ZerothOrderModelConfig
from . import optimizers, neural_networks as nn

DEFAULT_BATCH_SIZE = 50
DEFAULT_NB_EPOCHS = 1

backprop_linear_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_linear_adam",
    id={},
    neural_network_config=nn.linear,
    optimizer_config=optimizers.first_order_sgd,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

backprop_linear_adam: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_linear_adam",
    id={},
    neural_network_config=nn.linear,
    optimizer_config=optimizers.first_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

backprop_xs_adam: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_xs_adam",
    id={},
    neural_network_config=nn.xs,
    optimizer_config=optimizers.first_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

backprop_xs_sgd: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_xs_sgd",
    id={},
    neural_network_config=nn.xs,
    optimizer_config=optimizers.first_order_sgd,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

backprop_s_adam_5epochs: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_s_adam_5epochs",
    id={},
    neural_network_config=nn.s,
    optimizer_config=optimizers.first_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=5,
)

backprop_linear_adam_5epochs: FirstOrderModelConfig = FirstOrderModelConfig(
    name="backprop_s_adam_5epochs",
    id={},
    neural_network_config=nn.linear,
    optimizer_config=optimizers.first_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=5,
)

multiplex_linear_adam: ZerothOrderModelConfig = ZerothOrderModelConfig(
    name="multiplex_linear_adam",
    id={},
    neural_network_config=nn.linear,
    gradient_estimator_config=optimizers.simultaneous_perturbation,
    optimizer_config=optimizers.zeroth_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

multiplex_linear_sgd: ZerothOrderModelConfig = ZerothOrderModelConfig(
    name="perturb_linear_sgd",
    id={},
    neural_network_config=nn.linear,
    gradient_estimator_config=optimizers.simultaneous_perturbation,
    optimizer_config=optimizers.zeroth_order_sgd,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)

finite_difference_linear_adam: ZerothOrderModelConfig = ZerothOrderModelConfig(
    name="multiplex_linear_adam",
    id={},
    neural_network_config=nn.linear,
    gradient_estimator_config=optimizers.finite_difference,
    optimizer_config=optimizers.zeroth_order_adam,
    loss=losses.CrossEntropy(),
    metric=Accuracy(),
    batch_size=DEFAULT_BATCH_SIZE,
    nb_epochs=DEFAULT_NB_EPOCHS,
)
