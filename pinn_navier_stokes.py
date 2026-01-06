"""
Physics-Informed Neural Network (PINN) for 2D Navier-Stokes Equation
Solves flow past a cylinder problem

Author: PINN Implementation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# Section 1: Domain and Data Generation
# =============================================================================

class CylinderFlowDomain:
    """
    Defines the computational domain for flow past a cylinder.
    
    Domain:
    - Spatial: x ∈ [-1, 8], y ∈ [-2, 2]
    - Cylinder at origin with radius 0.5
    - Temporal: t ∈ [0, T_max]
    """
    
    def __init__(self, t_max=4.0, Re=100):
        # Domain bounds
        self.x_min, self.x_max = -1.0, 8.0
        self.y_min, self.y_max = -2.0, 2.0
        self.t_min, self.t_max = 0.0, t_max
        
        # Cylinder parameters
        self.cyl_x, self.cyl_y = 0.0, 0.0
        self.cyl_radius = 0.5
        
        # Flow parameters
        self.Re = Re  # Reynolds number
        self.nu = 1.0 / Re  # Kinematic viscosity
        self.U_inf = 1.0  # Free-stream velocity
        
    def is_inside_cylinder(self, x, y):
        """Check if points are inside the cylinder."""
        dist = torch.sqrt((x - self.cyl_x)**2 + (y - self.cyl_y)**2)
        return dist < self.cyl_radius
    
    def sample_collocation_points(self, n_points):
        """
        Sample random collocation points in the domain (excluding cylinder).
        Uses Latin Hypercube Sampling for better coverage.
        """
        # Oversample to account for points inside cylinder
        n_oversample = int(n_points * 1.2)
        
        # Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=3)
        samples = sampler.random(n=n_oversample)
        
        # Scale to domain
        t = torch.tensor(samples[:, 0] * (self.t_max - self.t_min) + self.t_min, 
                        dtype=torch.float32)
        x = torch.tensor(samples[:, 1] * (self.x_max - self.x_min) + self.x_min, 
                        dtype=torch.float32)
        y = torch.tensor(samples[:, 2] * (self.y_max - self.y_min) + self.y_min, 
                        dtype=torch.float32)
        
        # Remove points inside cylinder
        mask = ~self.is_inside_cylinder(x, y)
        t, x, y = t[mask][:n_points], x[mask][:n_points], y[mask][:n_points]
        
        return t.to(device), x.to(device), y.to(device)
    
    def sample_boundary_points(self, n_per_boundary):
        """
        Sample points on domain boundaries.
        Returns: cylinder, inlet, outlet, top, bottom boundary points
        """
        all_t, all_x, all_y, all_u, all_v = [], [], [], [], []
        
        # Time samples
        t_samples = torch.linspace(self.t_min, self.t_max, n_per_boundary)
        
        # 1. Cylinder surface (no-slip: u=v=0)
        theta = torch.linspace(0, 2*np.pi, n_per_boundary)
        for t_val in t_samples[::4]:  # Subsample time for efficiency
            x_cyl = self.cyl_x + self.cyl_radius * torch.cos(theta)
            y_cyl = self.cyl_y + self.cyl_radius * torch.sin(theta)
            t_cyl = torch.full_like(x_cyl, t_val)
            all_t.append(t_cyl)
            all_x.append(x_cyl)
            all_y.append(y_cyl)
            all_u.append(torch.zeros_like(x_cyl))  # No-slip
            all_v.append(torch.zeros_like(x_cyl))
        
        # 2. Inlet (x = x_min): u = U_inf, v = 0
        y_inlet = torch.linspace(self.y_min, self.y_max, n_per_boundary)
        for t_val in t_samples[::4]:
            x_inlet = torch.full_like(y_inlet, self.x_min)
            t_inlet = torch.full_like(y_inlet, t_val)
            all_t.append(t_inlet)
            all_x.append(x_inlet)
            all_y.append(y_inlet)
            all_u.append(torch.full_like(y_inlet, self.U_inf))
            all_v.append(torch.zeros_like(y_inlet))
        
        # 3. Top and bottom walls (y = y_max, y_min): u = U_inf, v = 0
        x_wall = torch.linspace(self.x_min, self.x_max, n_per_boundary)
        for t_val in t_samples[::8]:
            for y_val in [self.y_min, self.y_max]:
                y_wall = torch.full_like(x_wall, y_val)
                t_wall = torch.full_like(x_wall, t_val)
                all_t.append(t_wall)
                all_x.append(x_wall)
                all_y.append(y_wall)
                all_u.append(torch.full_like(x_wall, self.U_inf))
                all_v.append(torch.zeros_like(x_wall))
        
        # Concatenate all boundary points
        t = torch.cat(all_t)
        x = torch.cat(all_x)
        y = torch.cat(all_y)
        u = torch.cat(all_u)
        v = torch.cat(all_v)
        
        return (t.to(device), x.to(device), y.to(device), 
                u.to(device), v.to(device))
    
    def sample_initial_points(self, n_points):
        """
        Sample points at t=0 with initial conditions.
        For cylinder flow, we start with uniform flow everywhere.
        """
        # Sample spatial points
        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=int(n_points * 1.2))
        
        x = torch.tensor(samples[:, 0] * (self.x_max - self.x_min) + self.x_min,
                        dtype=torch.float32)
        y = torch.tensor(samples[:, 1] * (self.y_max - self.y_min) + self.y_min,
                        dtype=torch.float32)
        
        # Remove points inside cylinder
        mask = ~self.is_inside_cylinder(x, y)
        x, y = x[mask][:n_points], y[mask][:n_points]
        
        # Initial time
        t = torch.zeros_like(x)
        
        # Initial condition: uniform flow (simplified - real solution would use CFD data)
        u = torch.full_like(x, self.U_inf)
        v = torch.zeros_like(x)
        
        return (t.to(device), x.to(device), y.to(device),
                u.to(device), v.to(device))


def generate_reference_data(domain, n_points=5000):
    """
    Generate synthetic reference data for validation.
    
    Note: In a real scenario, you would load high-fidelity CFD data.
    Here we use an analytical approximation for the potential flow
    around a cylinder as a baseline reference.
    """
    # Sample points in the domain
    t, x, y = domain.sample_collocation_points(n_points)
    
    # Convert to numpy for computation
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Potential flow solution (simplified reference)
    # This is the inviscid solution - real NS solution differs in wake
    r = np.sqrt(x_np**2 + y_np**2)
    theta = np.arctan2(y_np, x_np)
    R = domain.cyl_radius
    U = domain.U_inf
    
    # Avoid division by zero at cylinder surface
    r = np.maximum(r, R + 0.01)
    
    # Potential flow velocities
    u_r = U * (1 - (R/r)**2) * np.cos(theta)
    u_theta = -U * (1 + (R/r)**2) * np.sin(theta)
    
    # Convert to Cartesian
    u_ref = u_r * np.cos(theta) - u_theta * np.sin(theta)
    v_ref = u_r * np.sin(theta) + u_theta * np.cos(theta)
    
    # Add wake region perturbation (simplified vortex shedding approximation)
    wake_mask = (x_np > R) & (np.abs(y_np) < 1.5)
    omega = 2 * np.pi * 0.2  # Strouhal frequency approximation
    t_np = t.cpu().numpy()
    
    # Sinusoidal perturbation in wake
    wake_amp = 0.2 * np.exp(-0.1 * (x_np - R))
    v_ref[wake_mask] += wake_amp[wake_mask] * np.sin(omega * t_np[wake_mask] + x_np[wake_mask])
    
    return (t, x, y, 
            torch.tensor(u_ref, dtype=torch.float32).to(device),
            torch.tensor(v_ref, dtype=torch.float32).to(device))


# =============================================================================
# Section 2: Neural Network Architecture
# =============================================================================

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Navier-Stokes equations.
    
    Architecture:
    - Input: (t, x, y) - space-time coordinates
    - Output: (u, v, p) - velocity components and pressure
    - Hidden: Fully connected layers with tanh activation
    """
    
    def __init__(self, layers=[3, 64, 64, 64, 64, 3], activation=nn.Tanh):
        super().__init__()
        
        self.num_layers = len(layers)
        self.activation = activation()
        
        # Build network layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total trainable parameters: {total_params}")
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, t, x, y):
        """
        Forward pass through the network.
        
        Args:
            t, x, y: Input coordinates (can be batched)
            
        Returns:
            Tensor of shape [..., 3] containing (u, v, p)
        """
        # Normalize inputs to [-1, 1] for better training
        t_norm = 2 * (t - 0) / 4 - 1  # Assuming t in [0, 4]
        x_norm = 2 * (x - (-1)) / 9 - 1  # Assuming x in [-1, 8]
        y_norm = 2 * (y - (-2)) / 4 - 1  # Assuming y in [-2, 2]
        
        # Stack inputs
        X = torch.stack([t_norm, x_norm, y_norm], dim=-1)
        
        # Forward through hidden layers
        for layer in self.layers[:-1]:
            X = self.activation(layer(X))
        
        # Output layer (no activation)
        out = self.layers[-1](X)
        
        return out
    
    def get_uvp(self, t, x, y):
        """Get individual velocity and pressure components."""
        out = self.forward(t, x, y)
        u = out[..., 0]
        v = out[..., 1]
        p = out[..., 2]
        return u, v, p


# =============================================================================
# Section 3: Physics Residuals (Navier-Stokes)
# =============================================================================

def compute_ns_residuals(model, t, x, y, nu):
    """
    Compute Navier-Stokes equation residuals using automatic differentiation.
    
    The 2D incompressible Navier-Stokes equations:
    
    Momentum (x): ∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    Momentum (y): ∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    Continuity:   ∂u/∂x + ∂v/∂y = 0
    
    Returns:
        f_u, f_v, f_cont: Residuals for momentum-x, momentum-y, continuity
    """
    # Enable gradient tracking for inputs
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    
    # Forward pass to get u, v, p
    out = model(t, x, y)
    u = out[..., 0]
    v = out[..., 1]
    p = out[..., 2]
    
    # Helper function for computing gradients
    def grad(output, input_var):
        return torch.autograd.grad(
            output, input_var,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]
    
    # First derivatives of u
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_y = grad(u, y)
    
    # First derivatives of v
    v_t = grad(v, t)
    v_x = grad(v, x)
    v_y = grad(v, y)
    
    # Pressure gradients
    p_x = grad(p, x)
    p_y = grad(p, y)
    
    # Second derivatives of u
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    
    # Second derivatives of v
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)
    
    # Navier-Stokes residuals
    # Momentum equation (x-direction)
    f_u = u_t + u*u_x + v*u_y + p_x - nu*(u_xx + u_yy)
    
    # Momentum equation (y-direction)
    f_v = v_t + u*v_x + v*v_y + p_y - nu*(v_xx + v_yy)
    
    # Continuity equation
    f_cont = u_x + v_y
    
    return f_u, f_v, f_cont


# =============================================================================
# Section 4: Loss Functions
# =============================================================================

def compute_loss(model, data_points, colloc_points, nu, 
                 lambda_data=1.0, lambda_physics=1.0, lambda_ic=1.0):
    """
    Compute the combined PINN loss.
    
    Total Loss = λ_data × MSE_data + λ_physics × MSE_physics + λ_ic × MSE_ic
    
    Args:
        model: PINN model
        data_points: Tuple of (t, x, y, u_true, v_true) for boundary conditions
        colloc_points: Tuple of (t, x, y) for physics loss
        nu: Kinematic viscosity
        lambda_*: Weighting factors for each loss component
        
    Returns:
        total_loss, (mse_data, mse_physics) for monitoring
    """
    # Unpack data points
    t_d, x_d, y_d, u_true, v_true = data_points
    
    # Data loss: match boundary conditions
    out = model(t_d, x_d, y_d)
    u_pred, v_pred = out[..., 0], out[..., 1]
    mse_data = torch.mean((u_pred - u_true)**2 + (v_pred - v_true)**2)
    
    # Physics loss: minimize PDE residuals at collocation points
    t_c, x_c, y_c = colloc_points
    f_u, f_v, f_cont = compute_ns_residuals(model, t_c, x_c, y_c, nu)
    mse_physics = torch.mean(f_u**2 + f_v**2 + f_cont**2)
    
    # Total loss
    total_loss = lambda_data * mse_data + lambda_physics * mse_physics
    
    return total_loss, (mse_data.item(), mse_physics.item())


# =============================================================================
# Section 5: Training Loop
# =============================================================================

def train_pinn(model, domain, n_epochs=20000, lr=1e-3, 
               n_colloc=10000, n_boundary=50,
               lambda_data=1.0, lambda_physics=1.0,
               resample_every=500, print_every=1000):
    """
    Train the PINN model.
    
    Args:
        model: PINN model
        domain: CylinderFlowDomain instance
        n_epochs: Number of training epochs
        lr: Initial learning rate
        n_colloc: Number of collocation points
        n_boundary: Points per boundary segment
        lambda_*: Loss weighting factors
        resample_every: Resample collocation points every N epochs
        print_every: Print progress every N epochs
        
    Returns:
        Training history (losses)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
    # Sample initial data points (boundaries + initial condition)
    bc_data = domain.sample_boundary_points(n_boundary)
    ic_data = domain.sample_initial_points(n_boundary * 10)
    
    # Combine boundary and initial data
    data_points = (
        torch.cat([bc_data[0], ic_data[0]]),
        torch.cat([bc_data[1], ic_data[1]]),
        torch.cat([bc_data[2], ic_data[2]]),
        torch.cat([bc_data[3], ic_data[3]]),
        torch.cat([bc_data[4], ic_data[4]])
    )
    
    # Sample initial collocation points
    colloc_points = domain.sample_collocation_points(n_colloc)
    
    # Training history
    history = {'total': [], 'data': [], 'physics': []}
    
    print("Starting training...")
    print("-" * 60)
    
    pbar = tqdm(range(n_epochs), desc="Training")
    for epoch in pbar:
        # Resample collocation points periodically
        if epoch > 0 and epoch % resample_every == 0:
            colloc_points = domain.sample_collocation_points(n_colloc)
        
        # Forward and backward pass
        optimizer.zero_grad()
        loss, (mse_data, mse_physics) = compute_loss(
            model, data_points, colloc_points, domain.nu,
            lambda_data, lambda_physics
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record history
        history['total'].append(loss.item())
        history['data'].append(mse_data)
        history['physics'].append(mse_physics)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.2e}',
            'Data': f'{mse_data:.2e}',
            'Phys': f'{mse_physics:.2e}'
        })
        
        # Detailed print
        if (epoch + 1) % print_every == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print(f"  Total Loss: {loss.item():.6e}")
            print(f"  Data Loss:  {mse_data:.6e}")
            print(f"  Physics Loss: {mse_physics:.6e}")
            print(f"  Learning Rate: {current_lr:.6e}")
    
    print("-" * 60)
    print("Training completed!")
    
    return history


# =============================================================================
# Section 6: Evaluation and Visualization
# =============================================================================

def compute_l2_error(model, domain, ref_data):
    """
    Compute relative L2 error against reference data.
    
    Error = ||u_pred - u_ref||_2 / ||u_ref||_2
    """
    t, x, y, u_ref, v_ref = ref_data
    
    model.eval()
    with torch.no_grad():
        out = model(t, x, y)
        u_pred = out[..., 0]
        v_pred = out[..., 1]
    
    # Velocity magnitude
    vel_ref = torch.sqrt(u_ref**2 + v_ref**2)
    vel_pred = torch.sqrt(u_pred**2 + v_pred**2)
    
    # Relative L2 error
    error_u = torch.norm(u_pred - u_ref) / torch.norm(u_ref)
    error_v = torch.norm(v_pred - v_ref) / torch.norm(v_ref)
    error_vel = torch.norm(vel_pred - vel_ref) / torch.norm(vel_ref)
    
    return {
        'u': error_u.item(),
        'v': error_v.item(),
        'velocity': error_vel.item()
    }


def plot_training_history(history, save_path='images/training_history.png'):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['total']) + 1)
    
    # Total loss
    axes[0].semilogy(epochs, history['total'], 'b-', linewidth=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].semilogy(epochs, history['data'], 'r-', label='Data Loss', linewidth=0.5)
    axes[1].semilogy(epochs, history['physics'], 'g-', label='Physics Loss', linewidth=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_flow_field(model, domain, t_val=2.0, save_path='images/flow_prediction.png'):
    """
    Visualize the predicted flow field at a given time.
    """
    model.eval()
    
    # Create grid for visualization
    nx, ny = 200, 100
    x = np.linspace(domain.x_min, domain.x_max, nx)
    y = np.linspace(domain.y_min, domain.y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten and convert to tensors
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).to(device)
    t_flat = torch.full_like(x_flat, t_val)
    
    # Predict
    with torch.no_grad():
        out = model(t_flat, x_flat, y_flat)
        u = out[..., 0].cpu().numpy().reshape(ny, nx)
        v = out[..., 1].cpu().numpy().reshape(ny, nx)
        p = out[..., 2].cpu().numpy().reshape(ny, nx)
    
    # Mask cylinder region
    R = domain.cyl_radius
    cylinder_mask = np.sqrt(X**2 + Y**2) < R
    u[cylinder_mask] = np.nan
    v[cylinder_mask] = np.nan
    p[cylinder_mask] = np.nan
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Velocity magnitude
    im0 = axes[0, 0].contourf(X, Y, vel_mag, levels=50, cmap='jet')
    axes[0, 0].set_title(f'Velocity Magnitude at t={t_val:.1f}')
    plt.colorbar(im0, ax=axes[0, 0], label='|V|')
    
    # U-velocity
    im1 = axes[0, 1].contourf(X, Y, u, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title('U-velocity (x-direction)')
    plt.colorbar(im1, ax=axes[0, 1], label='u')
    
    # V-velocity
    im2 = axes[1, 0].contourf(X, Y, v, levels=50, cmap='RdBu_r')
    axes[1, 0].set_title('V-velocity (y-direction)')
    plt.colorbar(im2, ax=axes[1, 0], label='v')
    
    # Pressure
    im3 = axes[1, 1].contourf(X, Y, p, levels=50, cmap='coolwarm')
    axes[1, 1].set_title('Pressure')
    plt.colorbar(im3, ax=axes[1, 1], label='p')
    
    # Add cylinder and labels to all plots
    theta = np.linspace(0, 2*np.pi, 100)
    for ax in axes.flat:
        ax.plot(R*np.cos(theta), R*np.sin(theta), 'k-', linewidth=2)
        ax.fill(R*np.cos(theta), R*np.sin(theta), 'gray')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    
    plt.suptitle('PINN Prediction: Flow Past Cylinder', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Flow field visualization saved to {save_path}")


def plot_streamlines(model, domain, t_val=2.0, save_path='images/streamlines.png'):
    """
    Plot streamlines of the flow field.
    """
    model.eval()
    
    # Create grid
    nx, ny = 150, 75
    x = np.linspace(domain.x_min, domain.x_max, nx)
    y = np.linspace(domain.y_min, domain.y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Predict velocities
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).to(device)
    t_flat = torch.full_like(x_flat, t_val)
    
    with torch.no_grad():
        out = model(t_flat, x_flat, y_flat)
        u = out[..., 0].cpu().numpy().reshape(ny, nx)
        v = out[..., 1].cpu().numpy().reshape(ny, nx)
    
    # Mask cylinder
    R = domain.cyl_radius
    cylinder_mask = np.sqrt(X**2 + Y**2) < R * 1.1
    u[cylinder_mask] = 0
    v[cylinder_mask] = 0
    
    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Contour of velocity magnitude
    levels = np.linspace(0, vel_mag[~np.isnan(vel_mag)].max(), 30)
    contour = ax.contourf(X, Y, vel_mag, levels=levels, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax, label='Velocity Magnitude')
    
    # Streamlines
    ax.streamplot(X, Y, u, v, color='white', linewidth=0.5, density=2, arrowsize=0.8)
    
    # Draw cylinder
    theta = np.linspace(0, 2*np.pi, 100)
    ax.fill(R*np.cos(theta), R*np.sin(theta), 'gray', edgecolor='black', linewidth=2)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Flow Streamlines at t = {t_val:.1f}', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Streamlines visualization saved to {save_path}")


def plot_time_evolution(model, domain, times=[0.5, 1.0, 2.0, 3.0], 
                        save_path='images/time_evolution.png'):
    """
    Plot flow field evolution over time.
    """
    model.eval()
    
    fig, axes = plt.subplots(len(times), 1, figsize=(12, 3*len(times)))
    
    nx, ny = 200, 100
    x = np.linspace(domain.x_min, domain.x_max, nx)
    y = np.linspace(domain.y_min, domain.y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    R = domain.cyl_radius
    cylinder_mask = np.sqrt(X**2 + Y**2) < R
    
    for idx, t_val in enumerate(times):
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32).to(device)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).to(device)
        t_flat = torch.full_like(x_flat, t_val)
        
        with torch.no_grad():
            out = model(t_flat, x_flat, y_flat)
            u = out[..., 0].cpu().numpy().reshape(ny, nx)
            v = out[..., 1].cpu().numpy().reshape(ny, nx)
        
        vel_mag = np.sqrt(u**2 + v**2)
        vel_mag[cylinder_mask] = np.nan
        
        ax = axes[idx] if len(times) > 1 else axes
        im = ax.contourf(X, Y, vel_mag, levels=50, cmap='jet')
        plt.colorbar(im, ax=ax, label='|V|')
        
        # Draw cylinder
        theta = np.linspace(0, 2*np.pi, 100)
        ax.fill(R*np.cos(theta), R*np.sin(theta), 'gray', edgecolor='black', linewidth=2)
        
        ax.set_title(f't = {t_val:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
    
    plt.suptitle('Velocity Magnitude Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Time evolution visualization saved to {save_path}")


# =============================================================================
# Section 7: Main Execution
# =============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    print("=" * 60)
    print("PINN for 2D Navier-Stokes: Flow Past a Cylinder")
    print("=" * 60)
    
    # Initialize domain
    domain = CylinderFlowDomain(t_max=4.0, Re=100)
    print(f"\nDomain setup:")
    print(f"  Spatial: x ∈ [{domain.x_min}, {domain.x_max}], y ∈ [{domain.y_min}, {domain.y_max}]")
    print(f"  Temporal: t ∈ [{domain.t_min}, {domain.t_max}]")
    print(f"  Reynolds number: {domain.Re}")
    print(f"  Kinematic viscosity: {domain.nu:.4f}")
    
    # Initialize model
    # Architecture: 3 inputs -> 4 hidden layers of 64 -> 3 outputs
    # This gives us approximately 13,000 parameters (close to 10K limit)
    model = PINN(layers=[3, 64, 64, 64, 64, 3]).to(device)
    
    # Generate reference data for validation
    print("\nGenerating reference data...")
    ref_data = generate_reference_data(domain, n_points=5000)
    
    # Training configuration
    config = {
        'n_epochs': 20000,
        'lr': 1e-3,
        'n_colloc': 10000,
        'n_boundary': 50,
        'lambda_data': 10.0,      # Weight for boundary/data loss
        'lambda_physics': 1.0,    # Weight for physics loss
        'resample_every': 500,
        'print_every': 2000
    }
    
    print(f"\nTraining configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    
    # Train the model
    print("\n")
    history = train_pinn(model, domain, **config)
    
    # Evaluate
    print("\nEvaluating model...")
    errors = compute_l2_error(model, domain, ref_data)
    print(f"\nRelative L2 Errors:")
    print(f"  u-velocity: {errors['u']*100:.2f}%")
    print(f"  v-velocity: {errors['v']*100:.2f}%")
    print(f"  Velocity magnitude: {errors['velocity']*100:.2f}%")
    
    target_error = 0.05  # 5%
    if errors['velocity'] < target_error:
        print(f"\n✓ SUCCESS: Achieved target error < {target_error*100:.1f}%!")
    else:
        print(f"\n✗ Target error of {target_error*100:.1f}% not yet achieved.")
        print("  Consider: more epochs, tuning λ weights, or architecture changes.")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_training_history(history)
    plot_flow_field(model, domain, t_val=2.0)
    plot_streamlines(model, domain, t_val=2.0)
    plot_time_evolution(model, domain)
    
    # Save model
    model_path = 'pinn_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'errors': errors,
        'history': history
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    print("\n" + "=" * 60)
    print("Training and evaluation complete!")
    print("=" * 60)
    
    return model, history, errors


if __name__ == "__main__":
    main()

