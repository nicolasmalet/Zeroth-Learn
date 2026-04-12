import zeroth.paths as paths
from zeroth.experiment import VariationConfig
from . import optimizers, neural_networks as nn

all_sizes = VariationConfig(
    name="Network Size",
    param=[paths.NN_CONFIG],
    values=[[nn.linear], [nn.xs], [nn.s], [nn.m], [nn.l], [nn.xl]]
)

small_networks = VariationConfig(
    name="Network Size",
    param=[paths.NN_CONFIG],
    values=[[nn.linear], [nn.xs], [nn.s]]
)

learning_rates_adam = VariationConfig(
    name="Learning Rate",
    param=[paths.LR],
    values=[[0.0001], [0.0003], [0.001], [0.003]]
)

learning_rates_sgd = VariationConfig(
    name="Learning Rate",
    param=[paths.LR],
    values=[[0.03], [0.1], [0.3], [1]]
)

optimizers_backprop = VariationConfig(
    name="Optimizer",
    param=[paths.OPTIMIZER_CONFIG],
    values=[[optimizers.first_order_adam], [optimizers.first_order_sgd]]
)

optimizers_spsa = VariationConfig(
    name="Optimizer",
    param=[paths.OPTIMIZER_CONFIG],
    values=[[optimizers.zeroth_order_adam], [optimizers.zeroth_order_sgd]]
)

nb_perturbations = VariationConfig(
    name="Nb Perturbations",
    param=[paths.NB_PERTURBATIONS],
    values=[[10], [30], [100]]
)

beta1 = VariationConfig(
    name="Beta 1",
    param=[paths.BETA1],
    values=[[0.9], [0.95], [0.99]]
)

beta2 = VariationConfig(
    name="Beta 2",
    param=[paths.BETA2],
    values=[[0.95], [0.99], [0.999]]
)

batch_sizes = VariationConfig(
    name="Batch Size",
    param=[paths.BATCH_SIZE],
    values=[[3], [10], [30]]
)
