from __future__ import annotations

from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .layer import Layer
from .neural_network import FirstOrderNeuralNetwork
from ..abstract.loss import Loss
from ..abstract.optimizer import Optimizer
from ..types import Array


@dataclass(frozen=True)
class FirstOrderOptimizerConfig(ABC):
    @abstractmethod
    def instantiate(self) -> FirstOrderOptimizer:
        pass


@dataclass(frozen=True)
class FirstOrderSGDConfig(FirstOrderOptimizerConfig):
    name = "SGD"
    learning_rate: float

    def instantiate(self) -> FirstOrderSGD:
        return FirstOrderSGD(self)


@dataclass(frozen=True)
class FirstOrderAdamConfig(FirstOrderSGDConfig):
    beta1: float
    beta2: float
    epsilon: float

    def instantiate(self) -> FirstOrderAdam:
        return FirstOrderAdam(self)


class FirstOrderOptimizer(Optimizer):
    @abstractmethod
    def do_descent(self, neural_network, loss, X: Array, Y_true: Array) -> float:
        pass


class FirstOrderSGD(FirstOrderOptimizer):
    """Abstract base class for gradient descent optimizers using first_order."""

    def __init__(self, config: FirstOrderSGDConfig) -> None:
        self.learning_rate: float = config.learning_rate

    def do_descent(self, neural_network: FirstOrderNeuralNetwork, loss: Loss, X: Array,
                   Y_true: Array) -> float:
        """Performs a full forward and backward pass for a single batch.

        1. Computes the output (Forward).
        2. Computes the loss and initial gradient dL/dZ (Backward init).
        3. Propagates gradients through all layers (Backpropagation).
        4. Updates weights and biases according to the specific optimizer rule.

        Args:
            neural_network (NeuralNetworkBackpropagation): The network to train.
            loss (Loss): The loss function to minimize.
            X (Array): The input data.
            Y_true (Array): The correct output.

        Returns:
            float: The average loss for the processed batch.
        """
        # 1. Forward Pass
        Y_pred = neural_network(X)
        last_layer = neural_network.layers[-1]

        avg_loss, dL_dZ = loss.compute_losses_for_first_order(last_layer, Y_pred, Y_true)

        dW = last_layer.X.T @ dL_dZ / last_layer.X.shape[0]
        dB = np.mean(dL_dZ, axis=0)

        # Propagation du gradient vers n-1
        dL_dAl = dL_dZ @ last_layer.W.T

        final_dW, final_dB = self._apply_update_rule(last_layer, dW, dB)
        last_layer.update_layer(final_dW, final_dB, self.learning_rate)

        for i in range(neural_network.nb_layers - 2, -1, -1):
            layer = neural_network.layers[i]
            dL_dAl, dW, dB = layer.get_gradient(dL_dAl)

            final_dW, final_dB = self._apply_update_rule(layer, dW, dB)
            layer.update_layer(final_dW, final_dB, self.learning_rate)

        return avg_loss

    def _apply_update_rule(self, layer: Layer, dW: Array, dB: Array) -> tuple[Array, Array]:
        return dW, dB


class FirstOrderAdam(FirstOrderSGD):
    name = "Adam"
    """Implements the Adam optimization algorithm.

    Adam (Adaptive Moment Estimation) stores moving averages of the gradients (m)
    and squared gradients (v) to adapt the learning rate for each parameter.
    """

    def __init__(self, config: FirstOrderAdamConfig) -> None:
        super().__init__(config)
        self.beta1: float = config.beta1
        self.beta2: float = config.beta2
        self.epsilon: float = config.epsilon

        self.beta1t: float = 1
        self.beta2t: float = 1
        self.m: dict[tuple[Layer, str], float | Array] = defaultdict(float)
        self.v: dict[tuple[Layer, str], float | Array] = defaultdict(float)

    def _apply_update_rule(self, layer: Layer, dW: Array, dB: Array) -> tuple[Array, Array]:
        """Computes the adaptive update step for a specific layer.

        Args:
            layer (Layer): The layer being updated (used as key for state dictionaries).
            dW (Array): Gradient w.r.t weights.
            dB (Array): Gradient w.r.t biases.

        Returns:
            tuple: The calculated updates (new_dW, new_dB) to be subtracted from params.
        """
        self.m[layer, "dW"] = self.beta1 * self.m[layer, "dW"] + (1 - self.beta1) * dW
        self.v[layer, "dW"] = self.beta2 * self.v[layer, "dW"] + (1 - self.beta2) * (dW ** 2)

        self.m[layer, "dB"] = self.beta1 * self.m[layer, "dB"] + (1 - self.beta1) * dB
        self.v[layer, "dB"] = self.beta2 * self.v[layer, "dB"] + (1 - self.beta2) * (dB ** 2)

        m_hat_w = self.m[layer, "dW"] / (1 - self.beta1t)
        v_hat_w = self.v[layer, "dW"] / (1 - self.beta2t)

        m_hat_b = self.m[layer, "dB"] / (1 - self.beta1t)
        v_hat_b = self.v[layer, "dB"] / (1 - self.beta2t)

        new_dW = m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        new_dB = m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        return new_dW, new_dB

    def do_descent(self, neural_network: FirstOrderNeuralNetwork, loss: Loss, X: Array,
                   Y_true: Array) -> float:
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        return super().do_descent(neural_network, loss, X, Y_true)
