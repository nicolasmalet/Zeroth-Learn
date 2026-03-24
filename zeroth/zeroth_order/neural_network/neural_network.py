from .parameter_manager import ParameterManager
from .. import GradientEstimator
from ..zeroth_order_blackbox import ZerothOrderBlackBox
from ...abstract.blackbox import NeuralNetworkConfig
from ...types import Array


class ZerothOrderNeuralNetwork(ZerothOrderBlackBox):
    """Neural Network implementation optimized for zeroth_order (parameter vector manipulation).

    Attributes:
        params (ParameterManager): Handler for flattening/reshaping weights (Theta <-> Ws/Bs).
    """

    def __init__(self, config: NeuralNetworkConfig) -> None:
        self.name: str = config.name
        self.params: ParameterManager = ParameterManager()
        self.input_dim: int = config.layers_config[0].input_dim
        self.output_dim: int = config.layers_config[-1].output_dim
        for layer_config in config.layers_config:
            self.params.push_layer(layer_config.output_dim,
                                   layer_config.input_dim,
                                   layer_config.f)

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
        for W, B, f in zip(self.params.Ws, self.params.Bs, self.params.fs):
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
        Ws, Bs = self.params.from_pThetas(pThetas)  # Ws: list of (T, in, out), Bs: list of (T, out)

        for W, B, f in zip(Ws, Bs, self.params.fs):
            X = X @ W + B
            X = f(X)

        return X

    def update_params(self, grad: Array, learning_rate: float) -> None:
        """Updates the flat parameter vector Theta and synchronizes Ws/Bs matrices."""
        self.params.Theta = self.params.Theta - learning_rate * grad
        self.params.update_weights_and_biases()
