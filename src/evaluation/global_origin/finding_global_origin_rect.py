import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import os

# cPro with Adam optimizer
class AdamCircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, stress, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.stress = stress
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

def circular_projection_adam(points, shift_point=None, lr=0.1, maxiter=100):
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]

    if shift_point is None:
        shift_point = torch.mean(points, dim=0)
    else:
        shift_point = torch.tensor(shift_point, dtype=torch.float32)
    points = points - shift_point

    hd_dist_mat = cosine_distances(points.detach().numpy())
    hd_dist_mat = torch.tensor(hd_dist_mat / 2, dtype=torch.float32)

    embedding = torch.randn(n, requires_grad=True)
    optimizer = torch.optim.Adam([embedding], lr=lr)
    loss_records = []

    def compute_ld_dist_matrix(ld_points):
        ld_points = ld_points.view(n, 1)
        dist_matrix = torch.cdist(ld_points, ld_points, p=1)
        return torch.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        ld_dist_mat = compute_ld_dist_matrix(ld_points)
        diff = torch.abs(hd_dist_mat - (2 * ld_dist_mat))
        return diff.sum() / 2

    for i in range(maxiter):
        optimizer.zero_grad()
        total_loss = loss(embedding)
        total_loss.backward()
        optimizer.step()
        loss_records.append(total_loss.item())

    ld_dist_matrix = compute_ld_dist_matrix(embedding).detach().numpy()
    stress = np.sqrt(np.sum((hd_dist_mat.numpy() - 2 * ld_dist_matrix) ** 2) / np.sum(hd_dist_mat.numpy() ** 2))

    return AdamCircularProjectionResult(
        embedding=embedding.detach().numpy(),
        circle_x=np.cos(embedding.detach().numpy() * 2 * np.pi),
        circle_y=np.sin(embedding.detach().numpy() * 2 * np.pi),
        loss_records=loss_records,
        stress=stress,
        hd_dist_matrix=hd_dist_mat.numpy(),
        ld_dist_matrix=ld_dist_matrix
    )

# Create square grid for stress evaluation
def create_square_grid(points, resolution=100, padding_ratio=0.05):
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    # Calculate the range for each dimension
    range_x = maxs[0] - mins[0]
    range_y = maxs[1] - mins[1]

    # Determine the maximum range to enforce square grid
    max_range = max(range_x, range_y)

    # Adjust padding for the square grid
    padding = max_range * padding_ratio
    center_x = (mins[0] + maxs[0]) / 2
    center_y = (mins[1] + maxs[1]) / 2

    # Create square bounds
    square_min_x = center_x - max_range / 2 - padding
    square_max_x = center_x + max_range / 2 + padding
    square_min_y = center_y - max_range / 2 - padding
    square_max_y = center_y + max_range / 2 + padding

    # Generate grid ranges
    x_range = np.linspace(square_min_x, square_max_x, resolution)
    y_range = np.linspace(square_min_y, square_max_y, resolution)

    # Create the grid
    grid = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
    return grid, x_range, y_range

def evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=100):
    stress_values = np.zeros(len(grid_points))
    for i, grid_point in enumerate(grid_points):
        print(f"Processing grid point {i + 1}/{len(grid_points)}...")
        try:
            result = circular_projection_adam(points, shift_point=grid_point, lr=lr, maxiter=maxiter)
            stress_values[i] = result.stress
        except Exception as e:
            print(f"[ERROR] Failed at grid point {grid_point}: {e}")
            stress_values[i] = np.nan
    return stress_values

def plot_stress_heatmap(points, stress_values, grid_x, grid_y, plot_output_file, heatmap_output_file):
    """
    Plots a square heatmap of stress values with equal pixel dimensions and highlights the three lowest stress points.
    """
    # Reshape stress values to grid format
    stress_grid = stress_values.reshape(len(grid_y), len(grid_x))

    # Apply logarithmic scaling for better visibility
    stress_grid_log = np.log10(1 + stress_grid - np.nanmin(stress_grid))

    # Clip stress values to the desired range [0, 0.3]
    stress_grid_clipped = np.clip(stress_grid_log, 0, 0.3)

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(8, 8))  # Ensure square figure
    heatmap = ax.imshow(
        stress_grid_clipped,
        extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]],
        origin='lower', cmap='viridis_r', alpha=1, vmin=0, vmax=0.08
    )

    # Add colorbar
    cbar = plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Log-Scaled Stress Level (0–0.3)")

    # Scatter the data points
    ax.scatter(points[:, 0], points[:, 1], c='orange', edgecolor='black', s=30, label='Data Points')

    # Calculate the grid points
    grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

    # Find the three lowest stress values
    lowest_indices = np.argsort(stress_values)[:3]
    lowest_points = np.array(grid_points)[lowest_indices]

    # Mark the three lowest stress points with white circles
    for i, point in enumerate(lowest_points):
        ax.scatter(
            point[0], point[1], facecolor='none', edgecolor='white', s=120, linewidth=2.5,
            label=f"Lowest Minima {i+1}" if i == 0 else None
        )

    # Ensure equal aspect ratio for the plot
    ax.set_aspect('equal', adjustable='box')

    # Title and labels
    ax.set_title("Stress Heatmap (Clipped to 0–0.3) with Equal Pixels")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Adjust the legend size and location
    legend = ax.legend(loc='upper right', frameon=True, fontsize='small', markerscale=1.2)
    legend.get_frame().set_edgecolor('black')  # Optional: Add black border to the legend box

    # Save the plot
    plt.savefig(plot_output_file, bbox_inches='tight')
    plt.show()

    # Save the clipped heatmap data for reuse
    np.save(heatmap_output_file, stress_grid_clipped)
    print(f"Clipped heatmap data exported to {heatmap_output_file}")



if __name__ == "__main__":
    from sklearn.datasets import load_iris

    # Load Iris dataset
    iris_data = load_iris()
    points = iris_data['data'][:, :2]  # Use only two dimensions for simplicity

    resolution = 150  # Adjust resolution to keep the computation feasible
    padding_ratio = 0.05  # Padding ratio for the extended grid
    stress_csv_file = os.path.join(os.getcwd(), "square_stress_data.csv")
    plot_output_file = os.path.join(os.getcwd(), "square_stress_heatmap_clipped.png")
    heatmap_output_file = os.path.join(os.getcwd(), "square_stress_heatmap_clipped.npy")

    # Create square grid
    grid_points, grid_x, grid_y = create_square_grid(points, resolution=resolution, padding_ratio=padding_ratio)

    # Load or evaluate stress values
    if os.path.exists(stress_csv_file):
        print(f"Loading precomputed stress data from {stress_csv_file}")
        stress_data = pd.read_csv(stress_csv_file)
        stress_values = stress_data['stress'].values

        # Validate stress_values size
        if len(stress_values) != len(grid_x) * len(grid_y):
            print("Mismatch detected! Recomputing stress values...")
            stress_values = evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=40)
            stress_data = pd.DataFrame(grid_points, columns=["dim_1", "dim_2"])
            stress_data["stress"] = stress_values
            stress_data.to_csv(stress_csv_file, index=False)
    else:
        print("Evaluating stress values on square grid...")
        stress_values = evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=50)
        stress_data = pd.DataFrame(grid_points, columns=["dim_1", "dim_2"])
        stress_data["stress"] = stress_values
        stress_data.to_csv(stress_csv_file, index=False)

    # Plot heatmap
    plot_stress_heatmap(points, stress_values, grid_x, grid_y, plot_output_file, heatmap_output_file)

    print(f"Square plot saved to {plot_output_file}")
