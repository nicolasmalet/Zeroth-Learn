from .parameter_manager import ParameterManager
from .. import GradientEstimator
from ..zeroth_order_blackbox import ZerothOrderBlackBox
from ...abstract import NeuralNetworkConfig, NeuralNetwork
from ...types import Array


class ZerothOrderNeuralNetwork(NeuralNetwork, ZerothOrderBlackBox):
    """Neural Network implementation optimized for zeroth_order (parameter vector manipulation).

    Attributes:
        params (ParameterManager): Handler for flattening/reshaping weights (Theta <-> Ws/Bs).
    """

    def __init__(self, config: NeuralNetworkConfig, input_dim: int, output_dim: int) -> None:
        super().__init__(config, input_dim, output_dim)
        self.params: ParameterManager = ParameterManager()

        network_dimensions = [input_dim] + config.architecture.hidden_dims + [output_dim]

        for i in range(self.nb_layers):
            self.params.push_layer(network_dimensions[i],
                                   network_dimensions[i + 1],
                                   config.architecture.activations[i])

    def __call__(self, X: Array) -> Array:
        return self.forward(X)

    def init_params(self, params: dict) -> None:
        self.params.Ws = params["Ws"]
        self.params.Bs = params["Bs"]
        self.params.update_theta()

    def get_params(self) -> dict:
        return {"Ws": self.params.Ws, "Bs": self.params.Bs}

    def forward(self, X: Array) -> Array:
        """Standard forward pass using the current nominal weights.

        Args:
            X (Array): Input batch. Shape: (batch_size, input_dim).

        Returns:
            Array: Output. Shape: (batch_size, output_dim).
        """
        for W, B, f in zip(self.params.Ws, self.params.Bs, self.params.fs, strict=True):
            X = f(X @ W + B)
        return X

    def forward_perturbed(self, X: Array, gradient_estimator: GradientEstimator) -> Array:
        """Parallel forward pass for multiple perturbed versions of the network.

        This method broadcasts the input X across T perturbed parameter sets
        to compute T outputs simultaneously without a Python loop.

        Args:
            X (Array): Input batch. Shape: (batch_size, input_dim).
            gradient_estimator (GradientEstimator): The gradient_estimator object.

        Returns:
            Array: Stacked outputs. Shape: (T, batch_size, output_dim)
                        where T is the number of perturbations.
        """
        pThetas = gradient_estimator.perturb(self.params.Theta)  # Shape: (T, nb_params)

        for (Ws, Bs), f in zip(self.params.iter_pThetas(pThetas), self.params.fs, strict=True):
            X = X @ Ws + Bs
            X = f(X)

        return X

    def update_params(self, grad: Array, learning_rate: float) -> None:
        """Updates the flat parameter vector Theta and synchronizes Ws/Bs matrices."""
        self.params.Theta = self.params.Theta - learning_rate * grad
        self.params.update_weights_and_biases()
