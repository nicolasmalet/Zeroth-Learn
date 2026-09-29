# Zeroth-Learn

**A NumPy library for estimating gradients from loss evaluations and training black-box models.**

The library implements coordinate finite differences, random-direction estimation, and SGD/Adam updates. Its MNIST experiments check that these components can train models when backpropagation is unavailable. The estimator and optimizer interfaces can also be used with other black boxes, including the simulator in [Quantum-Learn](https://github.com/nicolasmalet/Quantum-Learn).

## 1. Optimisation without derivatives

Let $\theta\in\mathbb R^d$ be model parameters and $f(\theta)$ a mini-batch loss. The model exposes values of $f$, but not $\nabla f$. Zeroth-order optimisation estimates a descent direction by evaluating nearby parameters:

```math
\theta_{k+1}=\theta_k-\eta_k\widehat{\nabla f}(\theta_k).
```

For Adam, the estimated gradient enters the usual first- and second-moment update in place of an exact gradient. No derivative of the black-box model is required.

## 2. Finite-difference estimators

A one-sided coordinate difference estimates component $i$ as

```math
\widehat{\partial_i f}(\theta)
=\frac{f(\theta+\delta e_i)-f(\theta)}{\delta}.
```

Estimating all $d$ components needs $d+1$ loss evaluations. [`GlobalFiniteDifference`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/gradient_estimators.py) does this for every coordinate; [`PartialFiniteDifference`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/gradient_estimators.py) evaluates only selected coordinates and returns zero elsewhere.

For a cheaper estimate in high dimension, [`SimultaneousPerturbation`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/gradient_estimators.py) draws $T$ Rademacher directions. At construction, the matrix $P\in\mathbb R^{T\times d}$ has entries $P_{ij}=s_{ij}/\sqrt T$, where each $s_{ij}$ is $+1$ or $-1$ with equal probability. The implementation uses

```math
\widehat{\nabla f}(\theta)
=\frac{1}{\delta}P^\top
\begin{pmatrix}
f(\theta+\delta P_1)-f(\theta)\\
\vdots\\
f(\theta+\delta P_T)-f(\theta)
\end{pmatrix}.
```

Because $\mathbb E[P^\top P]=I$, the first-order term recovers $\nabla f$ in expectation. Each update costs $T+1$ loss evaluations, independent of $d$ for a fixed $T$. Larger $T$ averages more directions but increases evaluation cost and memory.

## 3. From an estimator to a model update

The interfaces separate three operations:

| Operation                                                  | Implementation                                                              |
| ---------------------------------------------------------- | --------------------------------------------------------------------------- |
| Generate perturbed parameters and reconstruct the gradient | [`gradient_estimators.py`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/gradient_estimators.py)     |
| Generate Rademacher directions                             | [`perturbation_matrices.py`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/utils/perturbation_matrices.py)        |
| Evaluate a model at nominal and perturbed parameters       | [`zeroth_order_blackbox.py`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/zeroth_order_blackbox.py) |
| Apply SGD or Adam to the estimated gradient                | [`optimizers.py`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/optimizers.py)                       |

The included [`ZerothOrderNeuralNetwork`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/neural_network/neural_network.py) is one black-box implementation. [`ParameterManager`](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/zeroth/zeroth_order/neural_network/parameter_manager.py) maps weights and biases to a flat $\theta$; NumPy broadcasting evaluates its $T+1$ perturbed networks in one batch. This batching reduces Python overhead, not the number of loss evaluations. Other models choose how to perform those evaluations.

## 4. Numerical checks

A deterministic [test](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/tests/test_gradient_estimator.py) applies the random-direction estimator to a linear function, whose gradient is known. The MNIST experiments provide end-to-end checks of zeroth-order training on CPU. Their protocols, results, plots, and raw loss traces are in [Experiments](https://github.com/nicolasmalet/Zeroth-Learn/blob/main/EXPERIMENTS.md).

## 5. Reproduce the experiments

With [uv](https://docs.astral.sh/uv/), the lockfile fixes the dependency versions. The first experiment run downloads MNIST from OpenML.

```bash
git clone https://github.com/nicolasmalet/Zeroth-Learn.git
cd Zeroth-Learn
uv sync --locked --extra mnist
uv run --locked --extra mnist python -m unittest discover -s tests
MPLBACKEND=Agg uv run --locked --extra mnist python -m lab.mnist
MPLBACKEND=Agg uv run --locked --extra mnist python -m lab.mnist.run_experiment nb_perturbations_vs_model_size
```

The experiments write accuracy, loss traces, and plots under `results/`.

Without uv, install the package with its MNIST extra in a virtual environment, then run the same Python modules.

## 6. Scope

The MNIST experiments validate classical models. They are not benchmarks of quantum hardware or of PyTorch, and their saved plots do not contain repeated trials or uncertainty estimates. Memory use grows with the number of perturbations in the vectorized neural-network implementation.
