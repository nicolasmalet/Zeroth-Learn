from typing import Callable

import numpy as np

from ...types import Array, ActivationFunction
from ...utils.activation_functions import relu


class ParameterManager:
    """Manages the flattening and reshaping of neural network parameters for zeroth_order.

        In zeroth_order, we treat the entire network as a single vector Theta to apply perturbations.
        This class handles the mapping between:
        - The structured list of matrices (Ws, Bs) used for forward pass.
        - The flat vector (Theta) used for perturbation logic.

        Attributes:
            Ws (list[Array]): List of weight matrices for each layer.
            Bs (list[Array]): List of bias vectors.
            Theta (Array): The flattened parameter vector containing all Ws and Bs.
        """

    def __init__(self) -> None:
        self.Ws: list[Array] = []
        self.Bs: list[Array] = []
        self.fs: list[ActivationFunction] = []
        self.W_shapes: list[tuple] = []
        self.W_sizes: list[int] = []
        self.B_sizes: list[int] = []
        self.nb_layers: int = 0
        self.nb_params: int = 0
        self.Theta: Array = np.array([])

    def push_layer(self, output_dim: int, input_dim: int | None = None, f: Callable = relu) -> None:
        """Adds a layer to the structure and updates the flat Theta vector.

        Args:
            output_dim (int): Number of neurons in this layer.
            input_dim (int, optional): Input size. If None, inferred from previous layer.
            f (callable): Activation function.
        """
        input_dim = self.B_sizes[-1] if input_dim is None else input_dim
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        B = np.zeros(output_dim)
        self.Ws.append(W)
        self.Bs.append(B)
        self.fs.append(f)
        self.W_shapes.append(W.shape)
        self.W_sizes.append(W.size)
        self.B_sizes.append(output_dim)
        self.nb_layers += 1
        self.nb_params += W.size + B.size
        self.update_theta()

    def update_theta(self) -> None:
        """Re-builds the flat Theta vector from the current Ws and Bs matrices."""
        self.Theta = np.concatenate([W.ravel() for W in self.Ws] + [B.ravel() for B in self.Bs])

    def update_weights_and_biases(self) -> None:
        """Re-builds the Ws and Bs matrices from the current flat Theta vector."""

        self.Ws, self.Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            self.Ws.append(self.Theta[idx:idx + size].reshape(shape))
            idx += size
        for size in self.B_sizes:
            self.Bs.append(self.Theta[idx:idx + size])
            idx += size

    def from_pThetas(self, Thetas: Array) -> tuple[list[Array], list[Array]]:
        """Reconstructs temporary weight/bias matrices from a batch of perturbed Thetas.

        This is used to perform the forward pass on multiple perturbed models in parallel.

        Args:
            Thetas (Array): A batch of flat parameter vectors. 
                                 Shape: (nb_perturbations, nb_params)

        Returns:
            tuple: (Ws_list, Bs_list).
        """
        if Thetas.ndim == 1:
            Thetas = Thetas[None, :]
        N = Thetas.shape[0]

        Ws, Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            Ws.append(Thetas[:, idx:idx + size].reshape(N, *shape))
            idx += size
        for size in self.B_sizes:
            Bs.append(Thetas[:, idx:idx + size].reshape(N, 1, size))
            idx += size
        return Ws, Bs
