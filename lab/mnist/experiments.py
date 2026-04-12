from zeroth.experiment import ExperimentConfig
from . import variations, models
from .data import DataCreatorMnist

lr_vs_size_adam: ExperimentConfig = ExperimentConfig(name="lr_vs_size_adam",
                                                     base_model=models.backprop_linear_adam,
                                                     variations=[variations.small_networks,
                                                                 variations.learning_rates_adam],
                                                     data_creator=DataCreatorMnist())

lr_adam: ExperimentConfig = ExperimentConfig(name="lr_adam",
                                             base_model=models.backprop_linear_adam,
                                             variations=[variations.learning_rates_adam],
                                             data_creator=DataCreatorMnist())

lr_vs_size_sgd: ExperimentConfig = ExperimentConfig(name="lr_vs_size_sgd",
                                                    base_model=models.backprop_xs_sgd,
                                                    variations=[variations.small_networks,
                                                                variations.learning_rates_sgd],
                                                    data_creator=DataCreatorMnist())

small_sizes: ExperimentConfig = ExperimentConfig(name="small_sizes",
                                                 base_model=models.backprop_linear_adam,
                                                 variations=[variations.small_networks],
                                                 data_creator=DataCreatorMnist())

adam_vs_sgd: ExperimentConfig = ExperimentConfig(name="adam_vs_sgd",
                                                 base_model=models.backprop_s_adam_5epochs,
                                                 variations=[variations.optimizers_backprop,
                                                             variations.small_networks],
                                                 data_creator=DataCreatorMnist())

adam_betas: ExperimentConfig = ExperimentConfig(name="adam_betas",
                                                base_model=models.backprop_linear_adam,
                                                variations=[variations.beta1,
                                                            variations.beta2],
                                                data_creator=DataCreatorMnist())

linear_backprop: ExperimentConfig = ExperimentConfig(name="linear_backprop",
                                                     base_model=models.backprop_linear_adam,
                                                     variations=[],
                                                     data_creator=DataCreatorMnist())

batch_size: ExperimentConfig = ExperimentConfig(name="batch_size",
                                                base_model=models.backprop_linear_adam,
                                                variations=[variations.batch_sizes],
                                                data_creator=DataCreatorMnist())

nb_perturbations_vs_batch_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_batch_size",
                                                                    base_model=models.multiplex_linear_adam,
                                                                    variations=[variations.nb_perturbations,
                                                                                variations.batch_sizes],
                                                                    data_creator=DataCreatorMnist())

nb_perturbations_vs_model_size: ExperimentConfig = ExperimentConfig(name="nb_perturbations_vs_model_size",
                                                                    base_model=models.multiplex_linear_adam,
                                                                    variations=[variations.nb_perturbations,
                                                                                variations.small_networks],
                                                                    data_creator=DataCreatorMnist())

first_experiment: ExperimentConfig = ExperimentConfig(name="first_experiment",
                                                      base_model=models.backprop_linear_adam,
                                                      variations=[],
                                                      data_creator=DataCreatorMnist())

nb_perturbations: ExperimentConfig = ExperimentConfig(name="nb_perturbations",
                                                      base_model=models.multiplex_linear_adam,
                                                      variations=[variations.nb_perturbations],
                                                      data_creator=DataCreatorMnist())

nb_perturbations_adam_sgd: ExperimentConfig = ExperimentConfig(name="nb_perturbations_adam_sgd",
                                                               base_model=models.multiplex_linear_adam,
                                                               variations=[variations.nb_perturbations,
                                                                           variations.optimizers_spsa],
                                                               data_creator=DataCreatorMnist())
