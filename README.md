# Project

Implement a Physics-Informed Neural Network (PINN) in **PyTorch** to solve the **2D Navier-Stokes Equation** (Flow Past a Cylinder).

## Guide

### Compute Resources

* **For University students:** We will primarily use the university High-Performance Computing (HPC) cluster.

    1.  **Request an account:** Complete the cluster registration form via your department.
    2.  **Read the docs:** Review the generic SLURM guide or your cluster's specific wiki to understand GPU job submission.
    3.  We recommend using **NVIDIA A100 or V100** nodes for faster convergence during hyperparameter sweeps.

* **For external students:** You may use Google Colab (Pro recommended) or Kaggle Kernels.

    1.  Please ensure you enable GPU acceleration (T4 or P100) in your notebook settings.
    2.  While PINNs are less memory-hungry than DiTs, they are computationally intensive regarding gradient calculations (computing higher-order derivatives).

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
* The seminal paper: *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*.
* **Collocation Methods** and how to construct a loss function composed of a **Data Loss** ($MSE_u$) and a **Physics Loss** ($MSE_f$).

## Goal

Tune training hyperparameters (learning rate, weighting of loss terms, activation functions) and sampling strategies (distribution of collocation points) to obtain a **Relative $L_2$ Error lower than $5.0 \times 10^{-2}$ (5%)** on the velocity field $u(t, x, y)$.

Below is the target visualization: reconstruction of the wake dynamics behind the cylinder without using a mesh.


**More advanced:** Tweak the architecture to optimize convergence speed and accuracy, but **stay within 10,000 trainable parameters**. You might investigate techniques like Fourier Feature Embeddings or Adaptive Activation Functions to capture high-frequency turbulence.

## Instructions

* Create a private GitHub repository.
* Work independently.
* Feel free to refer to other resources or tutorials (e.g., DeepXDE logic), but you must write the PyTorch code yourself.
* **Implement the codebase yourself as much as possible**, specifically:
    * The `forward` pass that outputs $u, v, p$ (velocity and pressure).
    * The derivative calculation using `torch.autograd.grad` to construct the Navier-Stokes residuals (momentum and continuity equations).
    * The training loop.
* You may use AI or refer to others' code in an assisting capacity only, and you should be able to explain all the code.

## Weekly Reports

Submit a weekly report (within 3 pages each week, keep in the same Google Doc) at the end of each week.

* Feel free to structure it yourself: you can include progress (e.g., "Implemented boundary conditions"), issues / solutions, results, plots (heatmaps of the flow field), or negative results.
* **Crucial:** Plot the "Physics Loss" and "Data Loss" separately to monitor if the physics constraints are actually being learned.
* Please clearly indicate in which parts of the code you used others' code (link source) or AI, and in what capacity.

## Contact

Please send your weekly reports to the Instructor/TA email.
* You can schedule an initial meeting to discuss the formulation of the Navier-Stokes residuals if you are unfamiliar with fluid dynamics.
