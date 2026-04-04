import zeroth.utils.activation_functions as af
from zeroth.abstract import NetworkArchitecture, NeuralNetworkConfig


linear: NeuralNetworkConfig = NeuralNetworkConfig(
    name="Linear",
    architecture=NetworkArchitecture(hidden_dims=[],
                                     activations=[af.Softmax()])
)

xs: NeuralNetworkConfig = NeuralNetworkConfig(
    name="XS",
    architecture=NetworkArchitecture(hidden_dims=[128],
                                     activations=[af.ReLU(),
                                                 af.Softmax()])

)

s: NeuralNetworkConfig = NeuralNetworkConfig(
    name="S",
    architecture=NetworkArchitecture(hidden_dims=[128,
                                                  64],
                                     activations=[af.ReLU(),
                                                 af.ReLU(),
                                                 af.Softmax()])
)

m: NeuralNetworkConfig = NeuralNetworkConfig(
    name="M",
    architecture=NetworkArchitecture(hidden_dims=[256, 128,
                                                  64],
                                     activations=[af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.Softmax()])
)

l: NeuralNetworkConfig = NeuralNetworkConfig(
    name="L",
    architecture=NetworkArchitecture(hidden_dims=[256, 256, 128,
                                                  64],
                                     activations=[af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.Softmax()])
)
xl: NeuralNetworkConfig = NeuralNetworkConfig(
    name="XL",
    architecture=NetworkArchitecture(hidden_dims=[512, 512, 256, 128],
                                     activations=[af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.ReLU(),
                                                 af.Softmax()])

)
