import itertools
import os
from dataclasses import dataclass, replace
from typing import Callable, Union

import pandas as pd

from .abstract import Model, ModelConfig
from .data import Data
from .plot_losses import plot_losses
from .utils.dataclasses_utils import config_serializer, get_name, set_value_by_path


@dataclass(frozen=True)
class VariationConfig:
    name: str
    param: list[str]
    values: Union[list, list[list]]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    title: str
    base_model: ModelConfig
    variations: list[VariationConfig]
    create_data: Callable
    plot_dimension: int
    smooth_fraction: float

    def instantiate(self):
        return Experiment(self)


class Experiment:
    """Manages the full lifecycle of a deep learning experiment.

    It handles data loading, model instantiation, training loops, and results visualization.

    Attributes:
        name (str): Name of the experiment
        title (str): Title of the graphs
        models (list[Model]): List of models to train/compare
        data (Data): The dataset wrapper.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.name: str = config.name
        self.title: str = config.title
        self.base_model_config: ModelConfig = config.base_model
        self.models: list[Model] = generate_models(config.base_model, config.variations)
        self.data: Data = config.create_data()
        self.plot_dimension: int = config.plot_dimension
        self.smooth_fraction: float = config.smooth_fraction

        self.save_dir = os.path.join("experiments", self.name)

    def launch(self, do_train, do_test, nb_print_train, do_plot_train, do_save):
        """
        Executes the experiment pipeline.

        Args:
            do_train (bool): Whether to run the training loop.
            do_test (bool): Whether to run evaluation on test set.
            nb_print_train (int): Number of logs to print during training.
            do_plot_train (bool): If True, plots loss curves after training.
            do_save (bool): if True, saves the plots and dataframes
        """
        print(f"### Launching Experiment : {self.name} ###")
        if do_train:
            self.train(nb_print=nb_print_train, do_plot=do_plot_train, do_save=do_save)
        if do_test:
            self.test()
        if do_save:
            self.save_df()
            self.save_weights()

    def train(self, nb_print: int, do_plot: bool, do_save: bool):
        for model in self.models:
            model.train(self.data, nb_print)

        if do_plot:
            plot_path = None
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True)
                plot_path = os.path.join(self.save_dir, "training_losses.png")

            plot_losses(dimension=self.plot_dimension, models=self.models, title=self.title, smooth_fraction=self.smooth_fraction, save_path=plot_path)

    def test(self):
        for model in self.models:
            model.test(self.data)

    def save_df(self):
        """
        saves the models parameters and their args
        """
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"    Saving results to: {self.save_dir}")

        data = [model.id | {"test_loss": model.test_loss, "test_accuracy": model.test_accuracy}
                for model in self.models]

        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_dir, "models_accuracy.csv"), index_label="iteration")

    def save_weights(self) -> None:
        weights_dir = os.path.join(self.save_dir, "models")
        os.makedirs(weights_dir, exist_ok=True)
        print(f"    Saving weights to: {weights_dir}")

        for i, model in enumerate(self.models):
            model_dir = os.path.join(weights_dir, f"{model.get_folder_name()}")

            weights_path = os.path.join(model_dir, "weights.pkl")
            model.save_weights(weights_path)

            config_path = os.path.join(model_dir, "config.json")
            model.save_config(config_path)

            loss_path = os.path.join(model_dir, "training_loss.csv")
            model.save_loss(loss_path)


def generate_models(base_model: ModelConfig, variations: list[VariationConfig]) -> list[Model]:
    models = []

    values_lists = [v.values for v in variations]

    for combination in itertools.product(*values_lists):
        id_ = {}
        current_model = base_model

        for var_config, current_vals in zip(variations, combination):

            id_[var_config.name] = get_name(current_vals[0])

            for path, val in zip(var_config.param, current_vals):
                current_model = set_value_by_path(current_model, path, val)

        current_model = replace(current_model, id=id_)
        models.append(current_model.instantiate())

    return models