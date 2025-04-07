import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
import torch


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


def create_grid(points, resolution=100):
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    x_range = np.linspace(mins[0], maxs[0], resolution)
    y_range = np.linspace(mins[1], maxs[1], resolution)
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


def plot_stress_heatmap(points, stress_values, grid_points, grid_x, grid_y, resolution, plot_output_file, heatmap_output_file):
    stress_grid = stress_values.reshape((resolution, resolution))
    stress_grid_log = np.log10(1 + stress_grid - np.nanmin(stress_grid))
    stress_grid_clipped = np.clip(stress_grid_log, 0, 0.3)

    padding_x = (grid_x[-1] - grid_x[0]) * 0.02
    padding_y = (grid_y[-1] - grid_y[0]) * 0.02

    plt.figure(figsize=(10, 8))
    plt.imshow(
        stress_grid_clipped.T, extent=[
            grid_x[0] - padding_x, grid_x[-1] + padding_x,
            grid_y[0] - padding_y, grid_y[-1] + padding_y
        ],
        origin='lower', cmap='viridis_r', alpha=0.8, vmin=0, vmax=0.07
    )
    plt.colorbar(label="Log-Scaled Stress Level (0–0.07)")
    plt.scatter(points[:, 0], points[:, 1], c='orange', edgecolor='black', s=30, label='Data Points')

    lowest_indices = np.argsort(stress_values)[:3]
    lowest_points = grid_points[lowest_indices]

    for i, point in enumerate(lowest_points):
        plt.scatter(
            point[0], point[1], facecolor='none', edgecolor='white', s=120, linewidth=2.1,
            label=f"Lowest Minima" if i == 0 else None
        )

    plt.title("Stress Heatmap (Clipped to 0–0.3) with Data Points")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    legend = plt.legend(loc='upper right', frameon=True, fontsize='small', markerscale=1.2)
    legend.get_frame().set_edgecolor('black')

    plt.savefig(plot_output_file, bbox_inches='tight')
    plt.show()

    np.save(heatmap_output_file, stress_grid_clipped)
    print(f"Clipped heatmap data exported to {heatmap_output_file}")


def export_stress_data(grid_points, stress_values, output_file):
    stress_data = pd.DataFrame(grid_points, columns=['dim_1', 'dim_2'])
    stress_data['stress'] = stress_values
    stress_data.to_csv(output_file, index=False)
    print(f"Stress data exported to {output_file}")


def process_dimension_pairs(dataset, resolution=100, output_folder="heatmaps"):
    os.makedirs(output_folder, exist_ok=True)
    num_dimensions = dataset.shape[1]

    for i in range(num_dimensions):
        for j in range(i + 1, num_dimensions):
            print(f"Processing dimension pair: ({i + 1}, {j + 1})")
            points = dataset[:, [i, j]]

            mins, maxs = points.min(axis=0), points.max(axis=0)
            grid_x = np.linspace(mins[0], maxs[0], resolution)
            grid_y = np.linspace(mins[1], maxs[1], resolution)
            grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)

            stress_csv_file = os.path.join(output_folder, f"stress_data_dim_{i+1}_{j+1}.csv")
            plot_output_file = os.path.join(output_folder, f"stress_heatmap_dim_{i+1}_{j+1}.png")
            heatmap_output_file = os.path.join(output_folder, f"stress_heatmap_dim_{i+1}_{j+1}.npy")

            if os.path.exists(stress_csv_file):
                print(f"Loading precomputed stress data from {stress_csv_file}")
                stress_data = pd.read_csv(stress_csv_file)
                stress_values = stress_data['stress'].values
            else:
                print("Evaluating stress values...")
                stress_values = evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=50)
                export_stress_data(grid_points, stress_values, stress_csv_file)

            plot_stress_heatmap(points, stress_values, grid_points, grid_x, grid_y, resolution, plot_output_file, heatmap_output_file)
            print(f"Heatmap for dimension pair ({i + 1}, {j + 1}) saved to {plot_output_file}")


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    dataset = iris_data['data']
    process_dimension_pairs(dataset, resolution=100, output_folder="heatmaps")
