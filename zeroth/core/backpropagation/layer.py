import numpy as np

from zeroth.core.activation_functions import get_df


class Layer:
    """Represents a fully connected layer in a neural network using backpropagation.

    Attributes:
        W (np.ndarray): Weight matrix of shape (output_dim, input_dim).
        B (np.ndarray): Bias vector of shape (output_dim, 1).
        X (np.ndarray): Input stored during forward pass for backprop. Shape (input_dim, batch_size).
        Z (np.ndarray): Pre-activation linear combination. Shape (output_dim, batch_size).
        A (np.ndarray): Activated output. Shape (output_dim, batch_size).
    """

    def __init__(self, output_dim, input_dim, f):
        """Initializes the layer with random weights and zeros biases.

        Args:
            output_dim (int): Number of neurons in this layer.
            input_dim (int): Number of neurons in the previous layer.
            f (callable): Activation function (e.g., relu, sigmoid).
        """
        self.output_dim = output_dim
        self.f = f
        self.df = get_df[f]

        limit = np.sqrt(6.0 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        self.B = np.zeros((output_dim, 1))

        self.X = np.array([])
        self.Z = np.array([])
        self.A = np.array([])

    def get_output(self, X: np.ndarray) -> np.ndarray:
        """Performs the forward pass.

        Args:
            X (np.ndarray): Input data of shape (input_dim, batch_size).

        Returns:
            np.ndarray: The activated output (A) of shape (output_dim, batch_size).
        """
        self.X = X
        self.Z = np.matmul(self.W, X) + self.B
        self.A = self.f(self.Z)
        return self.A

    def get_gradient(self, dL_dA: np.ndarray):
        """Computes gradients during the backward pass.

        Args:
            dL_dA (np.ndarray): Gradient of the Loss w.r.t. the output A of this layer.
                                Shape: (output_dim, batch_size).

        Returns:
            tuple: A tuple containing:
                - dL_dA_prev (np.ndarray): Gradient to propagate to the previous layer (N-1).
                - dL_dW (np.ndarray): Gradient w.r.t. weights W.
                - dL_dB (np.ndarray): Gradient w.r.t. biases B.
        """
        df_Z = self.df(self.Z)
        dL_dZ = df_Z * dL_dA

        dL_dW = np.matmul(dL_dZ, self.X.T) / self.X.shape[1]
        dL_dB = np.mean(dL_dZ, axis=1, keepdims=True)

        dL_dA_prev = np.matmul(self.W.T, dL_dZ)

        return dL_dA_prev, dL_dW, dL_dB

    def update_layer(self, dW: np.ndarray, dB: np.ndarray, learning_rate: float):
        """Updates the parameters of the layer using the given gradients.

        Args:
            dW (np.ndarray): Calculated gradient for weights.
            dB (np.ndarray): Calculated gradient for biases.
            learning_rate (float): Step size for the update.
        """
        self.W = self.W - learning_rate * dW
        self.B = self.B - learning_rate * dB
