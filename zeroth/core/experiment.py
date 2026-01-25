from dataclasses import dataclass
from typing import Callable
import pandas as pd
import json
import os

from zeroth.core.dataclasses_utils import VariationConfig, generate_models, config_serializer
from zeroth.core.model import ModelConfig, Model
from zeroth.core.plot_losses import plot_losses


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    title: str
    base_model: ModelConfig
    variations: list[VariationConfig]
    create_data: Callable
    plot_dimension : int

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
        self.name = config.name
        self.title = config.title
        self.base_model = config.base_model
        self.models = generate_models(config.base_model, config.variations)
        self.data = config.create_data()
        self.plot_dimension = config.plot_dimension

        self.save_dir = os.path.join("results", self.name)

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

    def train(self, nb_print: int, do_plot: bool, do_save: bool):
        for model in self.models:
            model.train(self.data, nb_print)

        if do_plot:
            plot_path = None
            if do_save:
                os.makedirs(self.save_dir, exist_ok=True)
                plot_path = os.path.join(self.save_dir, "training_losses.png")

            plot_losses(dimension=self.plot_dimension, models=self.models, title=self.title, save_path=plot_path)

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

        config_path = os.path.join(self.save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, default=config_serializer, indent=4)
