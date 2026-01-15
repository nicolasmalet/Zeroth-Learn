# Zeroth-Learn

**Zeroth-Learn** is a research library I built to explore **Gradient-Free Optimization** in Machine Learning.

The primary goal of this project is to develop optimization techniques for training **Quantum Circuits**, where gradients are often costly or impossible to compute directly.

## Key Features

* **Hybrid Architecture:** Support for models trained via Backpropagation (Exact Gradient) and SPSA (Estimated Gradient).
* **Modular Engine:** Clean separation between optimization logic, model definitions, and experimental environments.
* **Quantum-Ready:** The SPSA implementation is specifically designed to simulate the constraints found in Noisy Intermediate-Scale Quantum (NISQ) processors.

## Experimental Results

To validate the library before deploying it on quantum simulations, I conducted a three-step benchmark on classical neural networks to determine the optimal stability conditions.

### 1. Hyperparameter Tuning: Finding the "Sweet Spot"
First, I determined the best model configuration and Learning Rate (LR) for both optimizers (Adam and SGD). I ran extensive sweeps to identify which learning rates offered the best convergence stability across different model depths.
* **Outcome:** Identified the critical learning rate thresholds where the optimizer stabilizes without diverging.

![Learning Rate Sensitivity](assets/plots/lr_vs_size_adam.png)
*(Fig 1. Impact of Learning Rate on loss across different network architectures.)*

### 2. Architecture Search: Determining Optimal Model Size
Once the hyperparameters were fixed, I evaluated the performance scalability. I trained networks of increasing depth (from Linear to Extra-Large) to find the trade-off between parameter complexity and convergence speed.

**Outcome:** The "Extra-Small" (XS) and "S" (S) architectures provided the best balance between expressivity and training stability for zeroth-order methods.

![Scalability Analysis](assets/plots/small_sizes.png)
*(Fig 2. Analysis of model depth vs. convergence efficiency.)*

### 3. SPSA Tuning: Optimizing the Perturbation Count
Finally, I tested the perturbative model (SPSA) to find the optimal `nb_perturbations` (samples per step). I measured how the gradient approximation quality improved as we increased the number of simultaneous perturbations.

**Outcome:** Increasing the perturbation count significantly reduces variance. A `nb_perturbations` value of 50 was found to be the most efficient compromise between computational cost and gradient fidelity.

![Gradient Estimation Analysis](assets/plots/nb_perturbations.png)
*(Fig 3. Effect of perturbation count on loss variance and convergence.)*
