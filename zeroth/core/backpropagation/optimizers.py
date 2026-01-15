from collections import defaultdict
from dataclasses import dataclass
from abc import abstractmethod
import numpy as np

from zeroth.core.common.optimizer import Optimizer
from zeroth.core.backpropagation.layer import Layer

@dataclass(frozen=True)
class OptimizerBackpropConfig:
    learning_rate: float

    def instantiate(self):
        return NotImplementedError


@dataclass(frozen=True)
class SGDBackpropagationConfig(OptimizerBackpropConfig):
    name = "SGD"
    def instantiate(self):
        return SGDBackpropagation(self)


@dataclass(frozen=True)
class AdamBackpropagationConfig(OptimizerBackpropConfig):

    name = "Adam"
    beta1: float
    beta2: float
    epsilon: float

    def instantiate(self):
        return AdamBackpropagation(self)


class OptimizerBackprop(Optimizer):
    """Abstract base class for gradient descent optimizers using backpropagation."""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def do_descent(self, neural_network, loss, data, batch_index):
        """Performs a full forward and backward pass for a single batch.

        1. Computes the output (Forward).
        2. Computes the loss and initial gradient dL/dZ (Backward init).
        3. Propagates gradients through all layers (Backpropagation).
        4. Updates weights and biases according to the specific optimizer rule.

        Args:
            neural_network (NeuralNetworkBackpropagation): The network to train.
            loss (Loss): The loss function to minimize.
            data (Data): The dataset handler.
            batch_index (int): The current batch index.

        Returns:
            float: The average loss for the processed batch.
        """
        # 1. Forward Pass
        X = data.X_train[batch_index]
        Y_pred = neural_network.get_output(X)
        last_layer = neural_network.layers[-1]

        avg_loss, dL_dZ = loss.get_backprop_init(last_layer, Y_pred, data, batch_index)

        m = last_layer.X.shape[1]
        dW = np.matmul(dL_dZ, last_layer.X.T) / m
        dB = np.mean(dL_dZ, axis=1, keepdims=True)

        # Propagation du gradient vers n-1
        dL_dAl = np.matmul(last_layer.W.T, dL_dZ)

        self._apply_and_update(last_layer, dW, dB)

        for i in range(neural_network.nb_layers - 2, -1, -1):
            layer = neural_network.layers[i]
            dL_dAl, dW, dB = layer.get_gradient(dL_dAl)
            self._apply_and_update(layer, dW, dB)

        self.end_of_batch()
        return avg_loss

    def _apply_and_update(self, layer, dW, dB):
        final_dW, final_dB = self.apply_update_rule(layer, dW, dB)
        layer.update_layer(final_dW, final_dB, self.learning_rate)

    @abstractmethod
    def apply_update_rule(self, layer, dW, dB):
        pass

    def end_of_batch(self):
        pass


class SGDBackpropagation(OptimizerBackprop):
    def __init__(self, config: SGDBackpropagationConfig):
        super().__init__(config.learning_rate)

    def apply_update_rule(self, layer, dW, dB):
        return dW, dB


class AdamBackpropagation(OptimizerBackprop):
    """Implements the Adam optimization algorithm.

    Adam (Adaptive Moment Estimation) stores moving averages of the gradients (m)
    and squared gradients (v) to adapt the learning rate for each parameter.
    """

    def __init__(self, config: AdamBackpropagationConfig):
        super().__init__(config.learning_rate)
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon

        self.beta1t = config.beta1
        self.beta2t = config.beta2
        self.m: dict[tuple[Layer, str], float | np.ndarray] = defaultdict(float)
        self.v: dict[tuple[Layer, str], float | np.ndarray] = defaultdict(float)

    def apply_update_rule(self, layer, dW, dB):
        """Computes the adaptive update step for a specific layer.

        Args:
            layer (Layer): The layer being updated (used as key for state dictionaries).
            dW (np.ndarray): Gradient w.r.t weights.
            dB (np.ndarray): Gradient w.r.t biases.

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

    def end_of_batch(self):
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
