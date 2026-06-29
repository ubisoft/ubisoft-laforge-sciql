import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def calculate_reward(position, center, radius, sigma):
    """Calculates the Gaussian reward for a given position."""
    center_dis = np.linalg.norm(position - center)
    error = center_dis - radius
    reward = np.exp(-(error**2) / (2 * sigma**2))
    return reward

def visualize_reward_landscape():
    """Generates and displays a heatmap of the reward function."""
    
    # --- Environment Parameters (from your Traj2D class) ---
    limits = np.array([[-50.0, 50.0], [-50.0, 50.0]])
    center_set = np.array([[0, 5], [0, 10], [0, -5], [0, -10]])
    radius_set = np.array([5, 10, 5, 10])
    
    # --- Visualization Settings ---
    mode_idx_to_visualize = 1  # The index of the target circle (e.g., 1 for center [0, 10], radius 10)
    sigma = 5.0                # The hyperparameter for the reward width
    grid_resolution = 400      # Number of points along each axis for the heatmap
    
    # --- Setup the Grid ---
    x_min, x_max = limits[0]
    y_min, y_max = limits[1]
    x_vals = np.linspace(x_min, x_max, grid_resolution)
    y_vals = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    
    # Get the target for the chosen mode
    target_center = center_set[mode_idx_to_visualize]
    target_radius = radius_set[mode_idx_to_visualize]

    # --- Calculate Reward for Every Point on the Grid ---
    reward_grid = np.zeros_like(xx)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            pos = np.array([xx[i, j], yy[i, j]])
            reward_grid[i, j] = calculate_reward(pos, target_center, target_radius, sigma)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create the heatmap
    c = ax.pcolormesh(xx, yy, reward_grid, cmap='viridis', shading='auto')
    fig.colorbar(c, ax=ax, label='Reward Value')
    
    # Overlay the target circle
    target_circle = Circle(
        target_center, target_radius,
        color='white', linestyle='--', fill=False,
        linewidth=2, label=f'Target Circle (Mode {mode_idx_to_visualize})'
    )
    ax.add_patch(target_circle)
    
    # Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Reward Landscape for Mode {mode_idx_to_visualize} (sigma = {sigma})")
    ax.legend()
    ax.grid(True, alpha=0.2)
    
    plt.show()

# Run the visualization
if __name__ == '__main__':
    visualize_reward_landscape()