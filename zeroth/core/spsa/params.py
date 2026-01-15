import numpy as np

from zeroth.core.activation_functions import relu


class Params:
    """Manages the flattening and reshaping of neural network parameters for spsa.

        In spsa, we treat the entire network as a single vector Theta to apply perturbations.
        This class handles the mapping between:
        - The structured list of matrices (Ws, Bs) used for forward pass.
        - The flat vector (Theta) used for perturbation logic.

        Attributes:
            Ws (list[np.ndarray]): List of weight matrices for each layer.
            Bs (list[np.ndarray]): List of bias vectors.
            Theta (np.ndarray): The flattened parameter vector containing all Ws and Bs.
        """
    def __init__(self):
        self.Ws, self.Bs, self.fs = [], [], []
        self.W_shapes, self.B_shapes = [], []
        self.W_sizes, self.B_sizes = [], []
        self.nb_layers, self.nb_params = 0, 0
        self.Theta = np.array([])

    def push_layer(self, output_dim, input_dim=None, f=relu):
        """Adds a layer to the structure and updates the flat Theta vector.

        Args:
            output_dim (int): Number of neurons in this layer.
            input_dim (int, optional): Input size. If None, inferred from previous layer.
            f (callable): Activation function.
        """
        input_dim = self.B_sizes[-1] if input_dim is None else input_dim
        limit = np.sqrt(6.0 / (input_dim + output_dim))
        W = np.random.uniform(-limit, limit, (output_dim, input_dim))
        B = np.zeros((output_dim, 1))
        self.Ws.append(W)
        self.Bs.append(B)
        self.fs.append(f)
        self.W_shapes.append(W.shape)
        self.B_shapes.append(B.shape)
        self.W_sizes.append(W.size)
        self.B_sizes.append(B.size)
        self.nb_layers += 1
        self.nb_params += W.size + B.size
        self.update_theta()

    def update_theta(self):
        """Re-builds the flat Theta vector from the current Ws and Bs matrices."""
        self.Theta = np.concatenate([W.ravel() for W in self.Ws] + [B.ravel() for B in self.Bs])

    def update_weights_and_biases(self):
        """Re-builds the Ws and Bs matrices from the current flat Theta vector."""

        self.Ws, self.Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            self.Ws.append(self.Theta[idx:idx + size].reshape(shape))
            idx += size
        for size, shape in zip(self.B_sizes, self.B_shapes):
            self.Bs.append(self.Theta[idx:idx + size].reshape(shape))
            idx += size

    def from_pThetas(self, Thetas):
        """Reconstructs temporary weight/bias matrices from a batch of perturbed Thetas.

        This is used to perform the forward pass on multiple perturbed models in parallel.

        Args:
            Thetas (np.ndarray): A batch of flat parameter vectors. 
                                 Shape: (nb_perturbations, nb_params)

        Returns:
            tuple: (Ws_list, Bs_list) where each element has shape (nb_perturbations, out, in).
        """
        if Thetas.ndim == 1:
            Thetas = Thetas[None, :]
        N = Thetas.shape[0]

        Ws, Bs = [], []
        idx = 0
        for size, shape in zip(self.W_sizes, self.W_shapes):
            Ws.append(Thetas[:, idx:idx + size].reshape(N, *shape))
            idx += size
        for size, shape in zip(self.B_sizes, self.B_shapes):
            Bs.append(Thetas[:, idx:idx + size].reshape(N, *shape))
            idx += size
        return Ws, Bs
