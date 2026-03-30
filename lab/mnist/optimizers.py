from zeroth import first_order as first
from zeroth import zeroth_order as zeroth
from zeroth.utils.perturbation_matrices import rademacher_matrix

DEFAULT_PERTURBATION_SCALE = 1e-8
DEFAULT_ADAM_LEARNING_RATE = 1e-3
DEFAULT_SGD_LEARNING_RATE = 0.1
DEFAULT_NB_PERTURBATION = 50

BETA1, BETA2 = 0.9, 0.99
EPSILON = 1e-8

first_order_sgd: first.FirstOrderSGDConfig = first.FirstOrderSGDConfig(learning_rate=DEFAULT_SGD_LEARNING_RATE)
first_order_adam: first.FirstOrderAdamConfig = first.FirstOrderAdamConfig(learning_rate=DEFAULT_ADAM_LEARNING_RATE,
                                                                          beta1=BETA1,
                                                                          beta2=BETA2,
                                                                          epsilon=EPSILON)

finite_difference: zeroth.GlobalFiniteDifferenceConfig = zeroth.GlobalFiniteDifferenceConfig(
    dA=DEFAULT_PERTURBATION_SCALE)
simultaneous_perturbation: zeroth.SimultaneousPerturbationConfig = zeroth.SimultaneousPerturbationConfig(
    dA=DEFAULT_PERTURBATION_SCALE,
    nb_perturbations=DEFAULT_NB_PERTURBATION,
    get_perturbation_matrix=rademacher_matrix)

zeroth_order_sgd: zeroth.ZerothOrderSGDConfig = zeroth.ZerothOrderSGDConfig(learning_rate=DEFAULT_SGD_LEARNING_RATE)
zeroth_order_adam: zeroth.ZerothOrderAdamConfig = zeroth.ZerothOrderAdamConfig(
    learning_rate=DEFAULT_ADAM_LEARNING_RATE,
    beta1=BETA1,
    beta2=BETA2,
    epsilon=EPSILON)
