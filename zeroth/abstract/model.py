from abc import ABC
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..plot_losses import plot_losses
from ..data import Data
from .optimizer import Optimizer
from .blackbox import BlackBox
from .loss import Loss



@dataclass(frozen=True)
class ModelConfig:
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
    nb_epochs: int

    def instantiate(self):
        pass

class Model(ABC):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the abstract logic for training
    regardless of the underlying engine (Backpropagation or zeroth_order).
    """

    def __init__(self, config: ModelConfig):

        self.name: str = config.name
        self.id: dict = config.id
        self.loss: Loss = config.loss
        self.metric: Callable = config.metric
        self.batch_size: int = config.batch_size
        self.nb_epochs: int = config.nb_epochs
        self.neural_network: BlackBox | None = None
        self.optimizer: Optimizer | None = None

        self.train_loss: np.ndarray = np.array([])
        self.test_loss: float | None = None
        self.test_accuracy: float | None = None

    def train(self, data: Data, nb_print: int = 0) -> None:
        """Runs the training loop over the dataset.

        Args:
            data (Data): The dataset object containing train/test sets.
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            np.ndarray: Array of loss values recorded at each step (for plotting).
        """

        nb_batches = data.nb_data // self.batch_size

        self.train_loss = np.zeros(self.nb_epochs * nb_batches, dtype=np.float64)

        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)
        print(f"    Training {self.id} Model")
        for epoch_idx in range(self.nb_epochs):
            print(f"        epoch n°{epoch_idx + 1} out of {self.nb_epochs}")
            data.prepare_data(self.batch_size)
            for batch_idx in range(nb_batches):
                X_train, Y_train = data.X_train[batch_idx], data.Y_train[batch_idx]
                avg_loss = self.optimizer.do_descent(self.neural_network, self.loss, X_train, Y_train)
                self.train_loss[epoch_idx * nb_batches + batch_idx] = avg_loss

                if batch_idx in print_indexes:
                    print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                          f"loss : {np.round(self.train_loss[epoch_idx * nb_batches + batch_idx], 3)}")
            self.test(data)

    def plot_loss(self) -> None:
        plot_losses(dimension=0, models=[self], title=self.name)


    def test(self, data: Data) -> None:
        X_test, Y_true = data.X_test, data.Y_test  # (in, batch), (out, batch)
        Y_pred = self.neural_network.forward(X_test)  # (out, batch)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")
