from zeroth.core.utils.dataclasses_utils import get_catalog_values
from zeroth.core.experiment import ExperimentConfig
from zeroth.core import VariationConfig

from .config import NETWORKS, OPTIMIZERS
from .data import create_data_mnist
from .models import MODELS

from dataclasses import dataclass


@dataclass(frozen=True)
class VariationCatalog:
    all_sizes = VariationConfig(param="neural_network_config",
                                values=get_catalog_values(NETWORKS))

    small_networks = VariationConfig(param="neural_network_config",
                                     values=[NETWORKS.Linear, NETWORKS.XS, NETWORKS.S])
    learning_rates_adam = VariationConfig(param="learning_rate", values=[0.0001, 0.0005, 0.001, 0.005])
    learning_rates_sgd = VariationConfig(param="learning_rate", values=[0.05, 0.1, 0.5, 1])
    optimizers_backprop = VariationConfig(param="optimizer_config",
                                          values=[OPTIMIZERS.FirstOrderAdam,
                                                  OPTIMIZERS.FirstOrderSGD])
    optimizers_spsa = VariationConfig(param="optimizer_config",
                                      values=[OPTIMIZERS.ZerothOrderAdam,
                                              OPTIMIZERS.ZerothOrderSGD])
    nb_perturbations = VariationConfig(param="nb_perturbations", values=[10, 30, 100])
    beta1 = VariationConfig(param="beta1", values=[0.9, 0.95, 0.99])
    beta2 = VariationConfig(param="beta2", values=[0.95, 0.99, 0.999])
    batch_sizes = VariationConfig(param="batch_size", values=[3, 10, 30])


VARIATIONS = VariationCatalog()


#  WARNING: some VariationConfig might overwrite others depending on the order of variations
@dataclass(frozen=True)
class ExperimentCatalog:
    first_experiment: ExperimentConfig = ExperimentConfig(name="first_experiment",
                                                          title="Congrats for your first experiment ! ",
                                                          base_model=MODELS.multiplex_linear_adam,
                                                          variations=[],
                                                          create_data=create_data_mnist,
                                                          plot_dimension=0)

    lr_vs_size_adam: ExperimentConfig = ExperimentConfig(name="lr_vs_size_adam",
                                                         title="Optimal Learning Rate across Model Depths with Adam",
                                                         base_model=MODELS.backprop_xs_adam,
                                                         variations=[VARIATIONS.small_networks,
                                                                     VARIATIONS.learning_rates_adam],
                                                         create_data=create_data_mnist,
                                                         plot_dimension=1)

    lr_adam: ExperimentConfig = ExperimentConfig(name="lr_adam",
                                                 title="Optimal Learning Rate with Adam optimizer",
                                                 base_model=MODELS.backprop_xs_adam,
                                                 variations=[VARIATIONS.learning_rates_adam],
                                                 create_data=create_data_mnist,
                                                 plot_dimension=0)

    lr_vs_size_sgd: ExperimentConfig = ExperimentConfig(name="lr_vs_size_sgd",
                                                        title="Optimal Learning Rate across Model Depths with SGD",
                                                        base_model=MODELS.backprop_xs_sgd,
                                                        variations=[VARIATIONS.small_networks,
                                                                    VARIATIONS.learning_rates_sgd],
                                                        create_data=create_data_mnist,
                                                        plot_dimension=2)

    all_sizes: ExperimentConfig = ExperimentConfig(name="all_sizes",
                                                   title="Effect of Network size",
                                                   base_model=MODELS.backprop_s_adam_5epochs,
                                                   variations=[VARIATIONS.all_sizes],
                                                   create_data=create_data_mnist,
                                                   plot_dimension=1)

    small_sizes: ExperimentConfig = ExperimentConfig(name="small_sizes",
                                                     title="Effect of Network size on loss",
                                                     base_model=MODELS.backprop_s_adam_5epochs,
                                                     variations=[VARIATIONS.small_networks],
                                                     create_data=create_data_mnist,
                                                     plot_dimension=1)

    adam_vs_sgd: ExperimentConfig = ExperimentConfig(name="adam_vs_sgd",
                                                     title="Adam vs GDP comparison",
                                                     base_model=MODELS.backprop_s_adam_5epochs,
                                                     variations=[VARIATIONS.optimizers_backprop,
                                                                 VARIATIONS.small_networks],
                                                     create_data=create_data_mnist,
                                                     plot_dimension=2)

    nb_perturbations_adam_sgd: ExperimentConfig = ExperimentConfig(name="nb_perturbations_adam_sgd",
                                                                   title="Effect of number of perturbations Adam vs SGD",
                                                                   base_model=MODELS.multiplex_linear_adam,
                                                                   variations=[VARIATIONS.nb_perturbations,
                                                                               VARIATIONS.optimizers_spsa],
                                                                   create_data=create_data_mnist,
                                                                   plot_dimension=1)

    adam_betas: ExperimentConfig = ExperimentConfig(name="adam_betas",
                                                    title="Adam Beta hyperparameters test",
                                                    base_model=MODELS.multiplex_linear_adam,
                                                    variations=[VARIATIONS.beta1,
                                                                VARIATIONS.beta2],
                                                    create_data=create_data_mnist,
                                                    plot_dimension=2)

    nb_perturbations: ExperimentConfig = ExperimentConfig(name="nb_perturbations",
                                                          title="Effect of Number of Perturbations on Training Loss",
                                                          base_model=MODELS.multiplex_linear_adam,
                                                          variations=[VARIATIONS.nb_perturbations],
                                                          create_data=create_data_mnist,
                                                          plot_dimension=0)

    linear_backprop: ExperimentConfig = ExperimentConfig(name="linear_backprop",
                                                         title="Linear Backpropagation Training Loss",
                                                         base_model=MODELS.backprop_linear_adam,
                                                         variations=[],
                                                         create_data=create_data_mnist,
                                                         plot_dimension=0)

    batch_size: ExperimentConfig = ExperimentConfig(name="batch_size",
                                                    title="Effect of Batch Sizes on Training Loss",
                                                    base_model=MODELS.backprop_linear_adam,
                                                    variations=[VARIATIONS.batch_sizes],
                                                    create_data=create_data_mnist,
                                                    plot_dimension=0)

    nb_perturbations_vs_batch_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_batch_size",
                                                                        title="Number of Perturbations vs Batch Sizes on Training Loss",
                                                                        base_model=MODELS.multiplex_linear_adam,
                                                                        variations=[VARIATIONS.nb_perturbations,
                                                                                    VARIATIONS.batch_sizes],
                                                                        create_data=create_data_mnist,
                                                                        plot_dimension=2)

    nb_perturbations_vs_model_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_model_size",
                                                                        title="Number of Perturbations vs Model Sizes on Training Loss",
                                                                        base_model=MODELS.multiplex_linear_adam,
                                                                        variations=[VARIATIONS.nb_perturbations,
                                                                                    VARIATIONS.small_networks],
                                                                        create_data=create_data_mnist,
                                                                        plot_dimension=1)


EXPERIMENTS = ExperimentCatalog()
