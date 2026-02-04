from dataclasses import dataclass
from matplotlib.axes import Axes
from typing import Callable
from abc import ABC

import pandas as pd
import numpy as np

from zeroth.core.zeroth_order import ZerothOrderNeuralNetwork, ZerothOrderOptimizerConfig, GradientEstimatorConfig, \
    GradientEstimator
from zeroth.core.first_order import FirstOrderNeuralNetwork, FirstOrderOptimizerConfig
from zeroth.core.abstract import NeuralNetwork, NeuralNetworkConfig, Optimizer
from zeroth.core.losses import Loss
from zeroth.core.data import Data


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
        raise NotImplementedError


@dataclass(frozen=True)
class FirstOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: FirstOrderOptimizerConfig

    def instantiate(self):
        return FirstOrderModel(self)


@dataclass(frozen=True)
class ZerothOrderModelConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: ZerothOrderOptimizerConfig
    gradient_estimator_config: GradientEstimatorConfig

    def instantiate(self):
        return ZerothOrderModel(self)


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
        self.neural_network: NeuralNetwork | None = None
        self.optimizer: Optimizer | None = None

        self.train_loss: np.ndarray = np.array([])
        self.test_loss: float | None = None
        self.test_accuracy: float | None = None

    def train(self, data: Data, nb_print: int = 0):
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

                avg_loss = self.optimizer.do_descent(self.neural_network, self.loss, data, batch_idx)
                self.train_loss[epoch_idx * nb_batches + batch_idx] = avg_loss

                if batch_idx in print_indexes:
                    print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                          f"loss : {np.round(self.train_loss[epoch_idx * nb_batches + batch_idx], 3)}")
            self.test(data)

    def plot_loss(self, ax: Axes, label: str, smooth_span: int = 50):
        ax.plot(self.train_loss, alpha=0.25, linewidth=1.0)
        smooth = self.smooth_curve(self.train_loss, smooth_span)
        ax.plot(smooth, label=label, linewidth=2.5)

    @staticmethod
    def smooth_curve(loss: np.ndarray, smooth_span: int) -> np.ndarray:
        return np.exp(pd.Series(np.log(loss)).ewm(span=smooth_span, adjust=True).mean())

    def test(self, data):
        X_test, Y_true = data.X_test, data.Y_test  # (in, batch), (out, batch)
        Y_pred = self.neural_network.forward(X_test)  # (out, batch)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.compute_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")


class FirstOrderModel(Model):
    def __init__(self, config: FirstOrderModelConfig):
        super().__init__(config)

        self.neural_network = FirstOrderNeuralNetwork(config.neural_network_config)
        self.optimizer = config.optimizer_config.instantiate()


class ZerothOrderModel(Model):
    def __init__(self, config: ZerothOrderModelConfig):
        super().__init__(config)

        self.neural_network: ZerothOrderNeuralNetwork = ZerothOrderNeuralNetwork(config.neural_network_config)
        nb_params = self.neural_network.params.nb_params
        self.gradient_estimator: GradientEstimator = config.gradient_estimator_config.instantiate(nb_params)
        self.optimizer: Optimizer = config.optimizer_config.instantiate(self.gradient_estimator)
