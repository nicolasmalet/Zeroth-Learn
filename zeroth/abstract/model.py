from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .blackbox import BlackBox
from .loss import Loss
from .optimizer import Optimizer
from .summary import Summary
from ..data import Data
from ..plot_losses import plot_losses
from ..types import Array


@dataclass(frozen=True, kw_only=True)
class ModelConfig(ABC, Summary):
    """
    name (str): Name of the model (used for display and saving).
    loss (Loss): The loss class.
    metric (Callable): Function (Y_pred, Y_true) -> score (e.g., accuracy).
    batch_size (int): Number of samples per gradient update.
    plot_results (Callable): Function to visualize test results.
    nb_epochs (int): Number of passes through the entire dataset.
    """
    name: str
    id: dict
    loss: Loss
    metric: Callable
    batch_size: int
    nb_epochs: int = 1

    @abstractmethod
    def instantiate(self, data: Data) -> Model:
        ...


class Model(ABC):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the abstract logic for training
    regardless of the underlying engine (Backpropagation or zeroth_order).
    """

    LOSS_FILE: str = "training_loss.csv"
    WEIGHTS_FILE: str = "weights.pkl"
    CONFIG_FILE: str = "config.json"

    neural_network: BlackBox
    optimizer: Optimizer

    def __init__(self, config: ModelConfig, data: Data):
        self.config = config

        self.name: str = config.name
        self.id: dict = config.id
        self.data: Data = data
        self.loss: Loss = config.loss
        self.metric: Callable = config.metric
        self.batch_size: int = config.batch_size
        self.nb_epochs: int = config.nb_epochs

        self.training_loss: Array = np.array([])
        self.test_loss: float = float("nan")
        self.test_accuracy: float = float("nan")

    def train(self, nb_print: int = 0) -> None:
        """Runs the training loop over the dataset.

        Args:
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            Array: Array of loss values recorded at each step (for plotting).
        """
        print(f"    Training {self.id} Model")

        self.data.batch_size = self.batch_size
        nb_batches = len(self.data)

        self.training_loss = np.zeros(self.nb_epochs * nb_batches, dtype=np.float64)

        nb_print = nb_batches if nb_print == -1 else nb_print
        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)

        for epoch_idx in range(self.nb_epochs):
            print(f"        epoch n°{epoch_idx + 1} out of {self.nb_epochs}")
            self.data.permutation()
            self.data.batch_size = self.batch_size

            for batch_idx, (X_train, Y_train) in enumerate(self.data):
                avg_loss = self.optimizer.do_descent(self.neural_network, self.loss, X_train, Y_train)
                self.training_loss[epoch_idx * nb_batches + batch_idx] = avg_loss

                if batch_idx in print_indexes:
                    print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                          f"loss : {np.round(self.training_loss[epoch_idx * nb_batches + batch_idx], 3)}")

            self.test()

    def plot_loss(self, smooth_fraction: float = 0.05) -> plt.Figure:
        fig = plot_losses(dimension=0,
                          models=[self],
                          title=self.name,
                          smooth_fraction=smooth_fraction)
        plt.close(fig)
        return fig

    def test(self) -> None:
        X_test, Y_true = self.data.X_test, self.data.Y_test  # (in, batch), (out, batch)
        Y_pred = self.neural_network(X_test)  # (out, batch)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")

    def save_loss(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / self.LOSS_FILE

        df = pd.DataFrame({
            'training_loss': self.training_loss
        })
        df.to_csv(save_path)

    def save_weights(self, save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / self.WEIGHTS_FILE

        params_dict = self.neural_network.get_params()
        with save_path.open('wb') as f:
            pickle.dump(params_dict, f)

    def load_weights(self, load_dir: Path) -> None:
        """Restaure les paramètres depuis un fichier pickle."""
        load_path = load_dir / self.WEIGHTS_FILE
        with load_path.open('rb') as f:
            params_dict = pickle.load(f)

        self.neural_network.init_params(params_dict)

    def get_folder_name(self) -> str:
        parts = []
        for key, value in self.id.items():
            clean_key = str(key).replace(" ", "_").lower()
            clean_val = str(value).replace(" ", "_").lower()
            parts.append(f"{clean_key}-{clean_val}")

        folder_name = "_".join(parts)

        return folder_name
