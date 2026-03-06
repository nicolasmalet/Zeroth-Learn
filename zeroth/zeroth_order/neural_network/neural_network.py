import numpy as np

from zeroth.abstract.blackbox import NeuralNetworkConfig
from zeroth.zeroth_order.zeroth_order_blackbox import ZerothOrderBlackBox
from zeroth.zeroth_order import GradientEstimator
from zeroth.zeroth_order.neural_network.parameter_manager import ParameterManager


class ZerothOrderNeuralNetwork(ZerothOrderBlackBox):
    """Neural Network implementation optimized for zeroth_order (parameter vector manipulation).

    Attributes:
        params (ParameterManager): Handler for flattening/reshaping weights (Theta <-> Ws/Bs).
    """
    def __init__(self, config: NeuralNetworkConfig):
        self.name: str = config.name
        self.params: ParameterManager = ParameterManager()
        for layer_config in config.layers_config:
            self.params.push_layer(layer_config.output_dim,
                                   layer_config.input_dim,
                                   layer_config.f)

    def init_params(self, params: tuple):
        Ws, Bs = params
        self.params.Ws = Ws
        self.params.Bs = Bs
        self.params.update_theta()

    def get_params(self) -> tuple:
        return self.params.Ws, self.params.Bs

    def forward(self, X):
        """Standard forward pass using the current nominal weights.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).

        Returns:
            np.ndarray: Output. Shape: (output_dim, batch_size).
        """
        for W, B, f in zip(self.params.Ws, self.params.Bs, self.params.fs):
            X = f(W @ X + B)
        return X

    def forward_perturbed(self, X: np.ndarray, gradient_estimator: GradientEstimator) -> np.ndarray:
        """Parallel forward pass for multiple perturbed versions of the network.

        This method broadcasts the input X across T perturbed parameter sets
        to compute T outputs simultaneously without a Python loop.

        Args:
            X (np.ndarray): Input batch. Shape: (input_dim, batch_size).
            gradient_estimator (GradientEstimator): The gradient_estimator object.

        Returns:
            np.ndarray: Stacked outputs. Shape: (T, output_dim, batch_size)
                        where T is the number of perturbations.
        """
        pThetas = gradient_estimator.perturb(self.params.Theta)  # Shape: (T, nb_params)
        Ws, Bs = self.params.from_pThetas(pThetas)  # Ws: list of (T, out, in), Bs: list of (T , out, 1)

        for W, B, f in zip(Ws, Bs, self.params.fs):
            X = W @ X + B
            X = f(X)

        return X

    def update_params(self, grad: np.ndarray, learning_rate: float) -> None:
        """Updates the flat parameter vector Theta and synchronizes Ws/Bs matrices."""
        self.params.Theta = self.params.Theta - learning_rate * grad
        self.params.update_weights_and_biases()

    def show_weights(self) -> str:
        return f"Ws, {self.params.Ws}, Bs {self.params.Bs}"
