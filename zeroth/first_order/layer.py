import numpy as np

from ..types import Array, ActivationFunction
from ..utils.activation_functions import get_df


class Layer:
    """Represents a fully connected layer in a neural network using first_order.

    Attributes:
        W (Array): Weight matrix of shape (input_dim, output_dim).
        B (Array): Bias vector of shape (output_dim).
        X (Array): Input stored during forward pass for backprop. Shape (batch_size, input_dim).
        Z (Array): Pre-activation linear combination. Shape (batch_size, output_dim).
        A (Array): Activated output. Shape (batch_size, output_dim).
    """

    def __init__(self, output_dim: int, input_dim: int, f: ActivationFunction) -> None:
        """Initializes the layer with random weights and zeros biases.

        Args:
            output_dim (int): Number of neurons in this layer.
            input_dim (int): Number of neurons in the previous layer.
            f (ActivationFunction): Activation function (e.g., relu, sigmoid).
        """
        self.output_dim: int = output_dim
        self.f: ActivationFunction = f
        self.df: ActivationFunction = get_df[f]

        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.W: Array = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.B: Array = np.zeros(output_dim)

        self.X: Array = np.array([])
        self.Z: Array = np.array([])
        self.A: Array = np.array([])

    def forward(self, X: Array) -> Array:
        """Performs the forward pass.

        Args:
            X (Array): Input data of shape (batch_size, input_dim).

        Returns:
            Array: The activated output (A) of shape (batch_size, output_dim).
        """
        self.X = X
        self.Z = X @ self.W + self.B
        self.A = self.f(self.Z)
        return self.A

    def get_gradient(self, dL_dA: Array) -> tuple[Array, Array, Array]:
        """Computes gradients during the backward pass.

        Args:
            dL_dA (Array): Gradient of the Loss w.r.t. the output A of this layer.
                                Shape: (batch_size, output_dim).

        Returns:
            tuple: A tuple containing:
                - dL_dA_prev (Array): Gradient to propagate to the previous layer (N-1).
                - dL_dW (Array): Gradient w.r.t. weights W.
                - dL_dB (Array): Gradient w.r.t. biases B.
        """
        df_Z = self.df(self.Z)
        dL_dZ = dL_dA * df_Z
        dL_dW = self.X.T @ dL_dZ / self.X.shape[0]
        dL_dB = np.mean(dL_dZ, axis=0)
        dL_dA_prev = dL_dZ @ self.W.T

        return dL_dA_prev, dL_dW, dL_dB

    def update_layer(self, dW: Array, dB: Array, learning_rate: float) -> None:
        """Updates the parameters of the layer using the given gradients.

        Args:
            dW (Array): Calculated gradient for weights.
            dB (Array): Calculated gradient for biases.
            learning_rate (float): Step size for the update.
        """
        self.W = self.W - learning_rate * dW
        self.B = self.B - learning_rate * dB
