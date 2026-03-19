from dataclasses import dataclass

import zeroth.paths as paths
from zeroth.experiment import ExperimentConfig
from zeroth.experiment import VariationConfig
from zeroth.utils.dataclasses_utils import get_catalog_values
from .configs import NETWORKS, OPTIMIZERS
from .data import create_data_mnist
from .models import MODELS


class VariationCatalog:
    all_sizes = VariationConfig(
        name="Network Size",
        param=[paths.NN_CONFIG],
        values=[[v] for v in get_catalog_values(NETWORKS)]
    )

    small_networks = VariationConfig(
        name="Network Size",
        param=[paths.NN_CONFIG],
        values=[[NETWORKS.linear], [NETWORKS.xs], [NETWORKS.s]]
    )

    learning_rates_adam = VariationConfig(
        name="Learning Rate",
        param=[paths.LR],
        values=[[0.0001], [0.0005], [0.001], [0.005]]
    )

    learning_rates_sgd = VariationConfig(
        name="Learning Rate",
        param=[paths.LR],
        values=[[0.05], [0.1], [0.5], [1]]
    )

    optimizers_backprop = VariationConfig(
        name="Optimizer",
        param=[paths.OPTIMIZER_CONFIG],
        values=[[OPTIMIZERS.first_order_adam], [OPTIMIZERS.first_order_sgd]]
    )

    optimizers_spsa = VariationConfig(
        name="Optimizer",
        param=[paths.OPTIMIZER_CONFIG],
        values=[[OPTIMIZERS.zeroth_order_adam], [OPTIMIZERS.zeroth_order_sgd]]
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


VARIATIONS = VariationCatalog()

SMOOTH_FRACTION = 0.05


#  WARNING: some VariationConfig might overwrite others depending on the order of variations
@dataclass(frozen=True)
class ExperimentCatalog:
    lr_vs_size_adam: ExperimentConfig = ExperimentConfig(name="lr_vs_size_adam",
                                                         title="Optimal Learning Rate across Model Depths with Adam",
                                                         base_model=MODELS.backprop_linear_adam,
                                                         variations=[VARIATIONS.small_networks,
                                                                     VARIATIONS.learning_rates_adam],
                                                         create_data=create_data_mnist,
                                                         plot_dimension=1,
                                                         smooth_fraction=SMOOTH_FRACTION)

    lr_adam: ExperimentConfig = ExperimentConfig(name="lr_adam",
                                                 title="Optimal Learning Rate with Adam optimizer",
                                                 base_model=MODELS.backprop_linear_adam,
                                                 variations=[VARIATIONS.learning_rates_adam],
                                                 create_data=create_data_mnist,
                                                 plot_dimension=0,
                                                 smooth_fraction=SMOOTH_FRACTION)

    lr_vs_size_sgd: ExperimentConfig = ExperimentConfig(name="lr_vs_size_sgd",
                                                        title="Optimal Learning Rate across Model Depths with SGD",
                                                        base_model=MODELS.backprop_xs_sgd,
                                                        variations=[VARIATIONS.small_networks,
                                                                    VARIATIONS.learning_rates_sgd],
                                                        create_data=create_data_mnist,
                                                        plot_dimension=2,
                                                        smooth_fraction=SMOOTH_FRACTION)

    small_sizes: ExperimentConfig = ExperimentConfig(name="small_sizes",
                                                     title="Effect of Network size on loss",
                                                     base_model=MODELS.backprop_s_adam_5epochs,
                                                     variations=[VARIATIONS.small_networks],
                                                     create_data=create_data_mnist,
                                                     plot_dimension=1,
                                                     smooth_fraction=SMOOTH_FRACTION)

    adam_vs_sgd: ExperimentConfig = ExperimentConfig(name="adam_vs_sgd",
                                                     title="Adam vs SGD comparison",
                                                     base_model=MODELS.backprop_s_adam_5epochs,
                                                     variations=[VARIATIONS.optimizers_backprop,
                                                                 VARIATIONS.small_networks],
                                                     create_data=create_data_mnist,
                                                     plot_dimension=2,
                                                     smooth_fraction=SMOOTH_FRACTION)

    adam_betas: ExperimentConfig = ExperimentConfig(name="adam_betas",
                                                    title="Adam Beta hyperparameters test",
                                                    base_model=MODELS.backprop_linear_adam,
                                                    variations=[VARIATIONS.beta1,
                                                                VARIATIONS.beta2],
                                                    create_data=create_data_mnist,
                                                    plot_dimension=2,
                                                    smooth_fraction=SMOOTH_FRACTION)

    linear_backprop: ExperimentConfig = ExperimentConfig(name="linear_backprop",
                                                         title="Linear Backpropagation Training Loss",
                                                         base_model=MODELS.backprop_linear_adam,
                                                         variations=[],
                                                         create_data=create_data_mnist,
                                                         plot_dimension=0,
                                                         smooth_fraction=SMOOTH_FRACTION)

    batch_size: ExperimentConfig = ExperimentConfig(name="batch_size",
                                                    title="Effect of Batch Sizes on Training Loss",
                                                    base_model=MODELS.backprop_linear_adam,
                                                    variations=[VARIATIONS.batch_sizes],
                                                    create_data=create_data_mnist,
                                                    plot_dimension=0,
                                                    smooth_fraction=SMOOTH_FRACTION)

    nb_perturbations_vs_batch_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_batch_size",
                                                                        title="Number of Perturbations vs Batch Sizes on Training Loss",
                                                                        base_model=MODELS.multiplex_linear_adam,
                                                                        variations=[VARIATIONS.nb_perturbations,
                                                                                    VARIATIONS.batch_sizes],
                                                                        create_data=create_data_mnist,
                                                                        plot_dimension=2,
                                                                        smooth_fraction=SMOOTH_FRACTION)

    nb_perturbations_vs_model_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_model_size",
                                                                        title="Number of Perturbations vs Model Sizes on Training Loss",
                                                                        base_model=MODELS.multiplex_linear_adam,
                                                                        variations=[VARIATIONS.nb_perturbations,
                                                                                    VARIATIONS.small_networks],
                                                                        create_data=create_data_mnist,
                                                                        plot_dimension=1,
                                                                        smooth_fraction=SMOOTH_FRACTION)

    first_experiment: ExperimentConfig = ExperimentConfig(name="first_experiment",
                                                          title="Congrats for your first experiment ! ",
                                                          base_model=MODELS.multiplex_linear_adam,
                                                          variations=[],
                                                          create_data=create_data_mnist,
                                                          plot_dimension=0,
                                                          smooth_fraction=SMOOTH_FRACTION)

    nb_perturbations: ExperimentConfig = ExperimentConfig(name="nb_perturbations",
                                                          title="Effect of Number of Perturbations on Training Loss",
                                                          base_model=MODELS.multiplex_linear_adam,
                                                          variations=[VARIATIONS.nb_perturbations],
                                                          create_data=create_data_mnist,
                                                          plot_dimension=0,
                                                          smooth_fraction=SMOOTH_FRACTION)

    nb_perturbations_adam_sgd: ExperimentConfig = ExperimentConfig(name="nb_perturbations_adam_sgd",
                                                                   title="Effect of number of perturbations Adam vs SGD",
                                                                   base_model=MODELS.multiplex_linear_adam,
                                                                   variations=[VARIATIONS.nb_perturbations,
                                                                               VARIATIONS.optimizers_spsa],
                                                                   create_data=create_data_mnist,
                                                                   plot_dimension=1,
                                                                   smooth_fraction=SMOOTH_FRACTION)


EXPERIMENTS = ExperimentCatalog()
