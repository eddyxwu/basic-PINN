# Project

Implement a Physics-Informed Neural Network (PINN) in **PyTorch** to solve the **2D Navier-Stokes Equation** (Flow Past a Cylinder).

<div align="center">
  <img src="images/flow_velocity.png" width="600" alt="Flow Velocity Visualization">
</div>

## Guide

### Compute Resources

* **For University students:** Feel free to use university resources as allowed.
* **For external students:** Try using Google Colab or Kaggle Kernels.

### PyTorch

You should use [PyTorch](https://pytorch.org/).
* Familiarize yourself specifically with `torch.autograd`, as computing gradients of the network output with respect to the *inputs* (spatial/temporal coordinates) is the core mechanism of a PINN.
* This [Automatic Differentiation tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) may be helpful.

### Dataset / Problem Domain

* **Problem:** 2D Navier-Stokes equation simulating flow past a circular cylinder.
* **Data:** You will not use "training data" in the traditional supervised sense for the entire domain. Instead, you will use:
    1.  **Collocation Points:** A set of randomly sampled coordinates $(t, x, y)$ inside the domain where the PDE residual is minimized.
    2.  **Boundary/Initial Data:** A small subset of exact data points at $t=0$ and on the simulation boundaries.
* **Validation:** Use the standard high-fidelity CFD reference solution to compute your error metrics.

### PINNs

You should understand the background knowledge:
* Neural Networks as functional approximators and Partial Differential Equations (PDEs).
* The seminal paper: [*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125).
* **Collocation Methods** and how to construct a loss function composed of a **Data Loss** ($MSE_u$) and a **Physics Loss** ($MSE_f$).

## Goal

Tune training hyperparameters (learning rate, weighting of loss terms, activation functions) and sampling strategies (distribution of collocation points) to obtain a **Relative $L_2$ Error lower than $5.0 \times 10^{-2}$ (5%)** on the velocity field $u(t, x, y)$.

Below is the target visualization: reconstruction of the wake dynamics behind the cylinder without using a mesh.


**More advanced:** Tweak the architecture to optimize convergence speed and accuracy, but **stay within 10,000 trainable parameters**. You might investigate techniques like Fourier Feature Embeddings or Adaptive Activation Functions to capture high-frequency turbulence.
