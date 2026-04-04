from zeroth.experiment import ExperimentConfig
from . import variations, models
from .data import DataCreatorMnist

SMOOTH_FRACTION = 0.05

lr_vs_size_adam: ExperimentConfig = ExperimentConfig(name="lr_vs_size_adam",
                                                     title="Optimal Learning Rate across Model Depths with Adam",
                                                     base_model=models.backprop_linear_adam,
                                                     variations=[variations.small_networks,
                                                                 variations.learning_rates_adam],
                                                     data_creator=DataCreatorMnist(),
                                                     plot_dimension=1,
                                                     smooth_fraction=SMOOTH_FRACTION)

lr_adam: ExperimentConfig = ExperimentConfig(name="lr_adam",
                                             title="Optimal Learning Rate with Adam optimizer",
                                             base_model=models.backprop_linear_adam,
                                             variations=[variations.learning_rates_adam],
                                             data_creator=DataCreatorMnist(),
                                             plot_dimension=0,
                                             smooth_fraction=SMOOTH_FRACTION)

lr_vs_size_sgd: ExperimentConfig = ExperimentConfig(name="lr_vs_size_sgd",
                                                    title="Optimal Learning Rate across Model Depths with SGD",
                                                    base_model=models.backprop_xs_sgd,
                                                    variations=[variations.small_networks,
                                                                variations.learning_rates_sgd],
                                                    data_creator=DataCreatorMnist(),
                                                    plot_dimension=2,
                                                    smooth_fraction=SMOOTH_FRACTION)

small_sizes: ExperimentConfig = ExperimentConfig(name="small_sizes",
                                                 title="Effect of Network size on loss",
                                                 base_model=models.backprop_linear_adam,
                                                 variations=[variations.small_networks],
                                                 data_creator=DataCreatorMnist(),
                                                 plot_dimension=0,
                                                 smooth_fraction=SMOOTH_FRACTION)

adam_vs_sgd: ExperimentConfig = ExperimentConfig(name="adam_vs_sgd",
                                                 title="Adam vs SGD comparison",
                                                 base_model=models.backprop_s_adam_5epochs,
                                                 variations=[variations.optimizers_backprop,
                                                             variations.small_networks],
                                                 data_creator=DataCreatorMnist(),
                                                 plot_dimension=2,
                                                 smooth_fraction=SMOOTH_FRACTION)

adam_betas: ExperimentConfig = ExperimentConfig(name="adam_betas",
                                                title="Adam Beta hyperparameters test",
                                                base_model=models.backprop_linear_adam,
                                                variations=[variations.beta1,
                                                            variations.beta2],
                                                data_creator=DataCreatorMnist(),
                                                plot_dimension=2,
                                                smooth_fraction=SMOOTH_FRACTION)

linear_backprop: ExperimentConfig = ExperimentConfig(name="linear_backprop",
                                                     title="Linear Backpropagation Training Loss",
                                                     base_model=models.backprop_linear_adam,
                                                     variations=[],
                                                     data_creator=DataCreatorMnist(),
                                                     plot_dimension=0,
                                                     smooth_fraction=SMOOTH_FRACTION)

batch_size: ExperimentConfig = ExperimentConfig(name="batch_size",
                                                title="Effect of Batch Sizes on Training Loss",
                                                base_model=models.backprop_linear_adam,
                                                variations=[variations.batch_sizes],
                                                data_creator=DataCreatorMnist(),
                                                plot_dimension=0,
                                                smooth_fraction=SMOOTH_FRACTION)

nb_perturbations_vs_batch_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_batch_size",
                                                                    title="Number of Perturbations vs Batch Sizes on Training Loss",
                                                                    base_model=models.multiplex_linear_adam,
                                                                    variations=[variations.nb_perturbations,
                                                                                variations.batch_sizes],
                                                                    data_creator=DataCreatorMnist(),
                                                                    plot_dimension=2,
                                                                    smooth_fraction=SMOOTH_FRACTION)

nb_perturbations_vs_model_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_model_size",
                                                                    title="Number of Perturbations vs Model Sizes on Training Loss",
                                                                    base_model=models.multiplex_linear_adam,
                                                                    variations=[variations.nb_perturbations,
                                                                                variations.small_networks],
                                                                    data_creator=DataCreatorMnist(),
                                                                    plot_dimension=1,
                                                                    smooth_fraction=SMOOTH_FRACTION)

first_experiment: ExperimentConfig = ExperimentConfig(name="first_experiment",
                                                      title="Congrats for your first experiment ! ",
                                                      base_model=models.backprop_linear_adam,
                                                      variations=[],
                                                      data_creator=DataCreatorMnist(),
                                                      plot_dimension=0,
                                                      smooth_fraction=SMOOTH_FRACTION)

nb_perturbations: ExperimentConfig = ExperimentConfig(name="nb_perturbations",
                                                      title="Effect of Number of Perturbations on Training Loss",
                                                      base_model=models.multiplex_linear_adam,
                                                      variations=[variations.nb_perturbations],
                                                      data_creator=DataCreatorMnist(),
                                                      plot_dimension=0,
                                                      smooth_fraction=SMOOTH_FRACTION)

nb_perturbations_adam_sgd: ExperimentConfig = ExperimentConfig(name="nb_perturbations_adam_sgd",
                                                               title="Effect of number of perturbations Adam vs SGD",
                                                               base_model=models.multiplex_linear_adam,
                                                               variations=[variations.nb_perturbations,
                                                                           variations.optimizers_spsa],
                                                               data_creator=DataCreatorMnist(),
                                                               plot_dimension=1,
                                                               smooth_fraction=SMOOTH_FRACTION)
