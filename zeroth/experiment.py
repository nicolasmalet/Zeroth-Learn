from __future__ import annotations

import itertools
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from .abstract import Model, ModelConfig, DataCreator, Summary
from .data import Data
from .plot_losses import plot_losses
from .utils.dataclasses_utils import get_name, set_value_by_path


@dataclass(frozen=True)
class VariationConfig:
    name: str
    param: list[str]
    values: Union[list, list[list]]


@dataclass(frozen=True)
class ExperimentConfig(Summary):
    name: str
    base_model: ModelConfig
    data_creator: DataCreator
    variations: list[VariationConfig]

    def instantiate(self) -> Experiment:
        return Experiment(self)


class Experiment:
    """Manages the full lifecycle of a deep learning experiment.

    It handles data loading, model instantiation, training loops, and results visualization.

    Attributes:
        name (str): Name of the experiment
        models (list[Model]): List of models to train/compare
        data (Data): The dataset wrapper.
    """

    ACCURACY_FILE: str = "models_accuracy.csv"
    CONFIG_FILE: str = "config.txt"

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.name: str = config.name
        self.base_model_config: ModelConfig = config.base_model
        self.data = config.data_creator()
        self.models: list[Model] = generate_models(config.base_model, config.variations, self.data)

    def train_models(self, nb_print: int) -> None:
        print(f"Training Models")
        for model in self.models:
            model.train(nb_print)

    def plot_losses(self, title: str, plot_dimension: int, smooth_fraction: float = 0) -> plt.Figure:
        fig = plot_losses(title=title,
                          dimension=plot_dimension,
                          models=self.models,
                          smooth_fraction=smooth_fraction)

        plt.close(fig)

        return fig

    def test_models(self) -> None:
        print(f"Testing Models")
        for model in self.models:
            model.test()

    def save_df(self, save_dir: Path) -> None:
        """
        saves the models parameters and their args
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / self.ACCURACY_FILE
        print(f"    Saving results to: {save_dir}")

        data = [model.id | {"test_loss": model.test_loss, "test_accuracy": model.test_accuracy}
                for model in self.models]

        df = pd.DataFrame(data)
        df.to_csv(save_path)

    def save_weights(self, save_dir: Path) -> None:
        for i, model in enumerate(self.models):
            save_path = save_dir / model.name
            model.save_weights(save_path)

    def save_configs(self, save_dir: Path) -> None:

        config_path = save_dir / self.CONFIG_FILE
        self.config.save(config_path)

        for i, model in enumerate(self.models):
            save_path = save_dir / model.name
            model.config.save(save_path)


def generate_models(base_model: ModelConfig, variations: list[VariationConfig], data: Data) -> list[Model]:
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
        models.append(current_model.instantiate(data))

    return models
