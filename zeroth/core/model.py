from dataclasses import dataclass
from typing import Callable
from abc import ABC
import numpy as np

from zeroth.core.spsa import NeuralNetworkPerturbation, OptimizerPerturbationConfig, PerturbationConfig
from zeroth.core.backpropagation import NeuralNetworkBackpropagation, OptimizerBackpropConfig
from zeroth.core.common import NeuralNetworkConfig, Loss


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
class ModelBackpropagationConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: OptimizerBackpropConfig

    def instantiate(self):
        return ModelBackpropagation(self)


@dataclass(frozen=True)
class ModelPerturbationConfig(ModelConfig):
    neural_network_config: NeuralNetworkConfig
    optimizer_config: OptimizerPerturbationConfig
    perturbation_config: PerturbationConfig

    def instantiate(self):
        return ModelPerturbation(self)


class Model(ABC):
    """
    Base class orchestrating the training and testing loop.

    This class abstracts the common logic for training
    regardless of the underlying engine (Backpropagation or spsa).
    """

    def __init__(self, config: ModelConfig):

        self.name = config.name
        self.id = config.id
        self.loss = config.loss
        self.metric = config.metric
        self.batch_size = config.batch_size
        self.nb_epochs = config.nb_epochs
        self.neural_network = None
        self.optimizer = None

        self.train_loss = []
        self.test_loss = None
        self.test_accuracy = None

    def train(self, data, nb_print=0):
        """Runs the training loop over the dataset.

        Args:
            data (Data): The dataset object containing train/test sets.
            nb_print (int): Number of progress updates to print per epoch.

        Returns:
            np.ndarray: Array of loss values recorded at each step (for plotting).
        """

        nb_batches = data.nb_data // self.batch_size

        losses = np.zeros([self.nb_epochs * nb_batches])

        print_indexes = np.linspace(0, nb_batches - 1, nb_print).astype(int)
        print(f"    Training {self.id} Model")
        for epoch_idx in range(self.nb_epochs):
            print(f"        epoch n°{epoch_idx + 1} out of {self.nb_epochs}")
            data.prepare_data(self.batch_size)
            for batch_idx in range(nb_batches):

                avg_loss = self.optimizer.do_descent(self.neural_network, self.loss, data, batch_idx)
                losses[epoch_idx * nb_batches + batch_idx] = avg_loss

                if batch_idx in print_indexes:
                    print(f"            batch n°{batch_idx + 1} out of {nb_batches}, "
                          f"loss : {np.round(losses[epoch_idx * nb_batches + batch_idx], 3)}")
            self.test(data)
        self.train_loss += list(losses)

    def test(self, data):
        X_test, Y_true = data.X_test, data.Y_test  # (in, batch), (out, batch)
        Y_pred = self.neural_network.get_output(X_test)  # (out, batch)

        self.test_accuracy = self.metric(Y_pred, Y_true)
        self.test_loss = self.loss.get_avg_loss(Y_pred, Y_true)

        print(f"    {self.id} accuracy : {self.test_accuracy}, loss : {self.test_loss}")


class ModelBackpropagation(Model):
    def __init__(self, config: ModelBackpropagationConfig):
        super().__init__(config)

        self.neural_network = NeuralNetworkBackpropagation(config.neural_network_config)
        self.optimizer = config.optimizer_config.instantiate()


class ModelPerturbation(Model):
    def __init__(self, config: ModelPerturbationConfig):
        super().__init__(config)

        self.neural_network = NeuralNetworkPerturbation(config.neural_network_config)
        nb_params = self.neural_network.params.nb_params
        self.perturbation = config.perturbation_config.instantiate(nb_params)
        self.optimizer = config.optimizer_config.instantiate(self.perturbation)
