import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class AdamCircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, stress, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.stress = stress
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<AdamCircularProjectionResult 1-d-embedding={self.embedding}, stress={self.stress:.4f}>'

def circular_projection_adam(points, shift_point=None, lr=0.1, maxiter=100, max_time=None):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using Adam.
    Allows shifting the data to a custom point before projection.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    shift_point : array-like, shape (n_features,), optional (default=None)
        The point to which the data should be shifted. If None, the data is shifted to the center.

    lr : float, optional (default=0.1)
        The learning rate for the Adam optimizer.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    Returns:
    -------
    AdamCircularProjectionResult : object
        A result object containing the projection, stress metric, and metadata.
    """

    # Convert points to a numpy array if it's a DataFrame
    if isinstance(points, pd.DataFrame):
        points = points.values

    # Convert points to torch tensors for optimization
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]

    # Shift the data to the specified point or to the center
    if shift_point is None:
        shift_point = torch.mean(points, dim=0)  # Default: shift to center
    else:
        shift_point = torch.tensor(shift_point, dtype=torch.float32)  # Ensure tensor format
    points = points - shift_point

    # Calculate distances in the high-dimensional space using cosine distance
    hd_dist_mat = cosine_distances(points.detach().numpy())
    hd_dist_mat = torch.tensor(hd_dist_mat / 2, dtype=torch.float32)

    # Initialize the embedding as learnable parameters
    embedding = torch.randn(n, requires_grad=True)

    # Adam optimizer
    optimizer = torch.optim.Adam([embedding], lr=lr)

    # Record the loss during the iterations
    loss_records = []

    def compute_ld_dist_matrix(ld_points):
        """Compute the distances between points in the low-dimensional space."""
        ld_points = ld_points.view(n, 1)
        dist_matrix = torch.cdist(ld_points, ld_points, p=1)
        return torch.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        """Compute the difference between low-dimensional and high-dimensional distances."""
        ld_dist_mat = compute_ld_dist_matrix(ld_points)
        diff = torch.abs(hd_dist_mat - (2 * ld_dist_mat))
        return diff.sum() / 2

    # Start the timer for the time constraint
    start_time = time.time()

    # Optimization loop using Adam with a time constraint
    for i in range(maxiter):
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break

        optimizer.zero_grad()
        total_loss = loss(embedding)
        total_loss.backward()
        optimizer.step()
        loss_records.append(total_loss.item())

    # Convert the embedding to angles on the circle
    final_embedding = embedding.detach().numpy()
    circle_x = np.cos(final_embedding * 2 * np.pi)
    circle_y = np.sin(final_embedding * 2 * np.pi)

    # Compute the low-dimensional distance matrix for the final embedding
    ld_dist_matrix = compute_ld_dist_matrix(embedding).detach().numpy()

    # Compute the Kruskal stress
    stress = np.sqrt(np.sum((hd_dist_mat.numpy() - 2 * ld_dist_matrix) ** 2) / np.sum(hd_dist_mat.numpy() ** 2))

    return AdamCircularProjectionResult(
        embedding=final_embedding,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        stress=stress,
        hd_dist_matrix=hd_dist_mat.numpy(),
        ld_dist_matrix=ld_dist_matrix
    )

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import time
from itertools import product

# Define the AdamCircularProjectionResult class and circular_projection_adam function (use the earlier version).

def create_grid(points, resolution=5):
    """
    Creates a grid of points in the high-dimensional space based on the dataset's range.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points.

    resolution : int, optional (default=5)
        The number of grid points per dimension.

    Returns:
    -------
    grid_points : array, shape (n_grid_points, n_features)
        The grid points in the high-dimensional space.
    """
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    ranges = [np.linspace(mins[i], maxs[i], resolution) for i in range(points.shape[1])]
    grid_points = np.array(list(product(*ranges)))
    return grid_points

def evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=100):
    """
    Evaluates cPro for each grid point and collects the stress values.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points.

    grid_points : array-like, shape (n_grid_points, n_features)
        The grid points to shift the data to.

    lr : float, optional (default=0.1)
        The learning rate for the Adam optimizer.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    Returns:
    -------
    stress_results : dict
        A dictionary with grid points as keys and stress values as values.
    """
    stress_results = {}

    for i, grid_point in enumerate(grid_points):
        print(f"Processing grid point {i + 1}/{len(grid_points)}...")
        try:
            # Evaluate cPro with the grid point as the shift origin
            result = circular_projection_adam(points, shift_point=grid_point, lr=lr, maxiter=maxiter)
            # Store stress with grid point as key
            stress_results[tuple(grid_point)] = result.stress
        except Exception as e:
            print(f"[ERROR] Failed to process grid point {grid_point}: {e}")
            stress_results[tuple(grid_point)] = np.nan  # Record NaN for failed grid points

    return stress_results


def stress_to_dataframe(stress_results):
    """
    Converts stress results into a DataFrame for visualization.

    Parameters:
    ----------
    stress_results : dict
        A dictionary with grid points as keys and stress values as values.

    Returns:
    -------
    stress_df : pd.DataFrame
        A DataFrame containing the grid points and their corresponding stress values.
    """
    data = [list(key) + [value] for key, value in stress_results.items()]
    columns = [f"dim_{i+1}" for i in range(len(data[0]) - 1)] + ["stress"]
    stress_df = pd.DataFrame(data, columns=columns)
    return stress_df

def plot_scatter_matrix_with_stress(stress_df):
    """
    Creates a scatterplot matrix with a color scale based on stress values.

    Parameters:
    ----------
    stress_df : pd.DataFrame
        A DataFrame containing the grid points and their corresponding stress values.
    """
    # Normalize stress for color mapping
    stress_df['normalized_stress'] = (stress_df['stress'] - stress_df['stress'].min()) / \
                                     (stress_df['stress'].max() - stress_df['stress'].min())

    # Pairplot with color-coded stress
    pairplot = sns.pairplot(
        stress_df,
        vars=stress_df.columns[:-2],  # Exclude 'stress' and 'normalized_stress' from axes
        hue='normalized_stress',
        palette='viridis',
        diag_kind="kde",
        plot_kws={"s": 50, "alpha": 0.8}  # Customize scatterplot points
    )
    pairplot.fig.suptitle("Scatterplot Matrix with Stress-Based Coloring", y=1.02)
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), label='Normalized Stress')

    # Highlight the best origin position
    best_point = stress_df.loc[stress_df['stress'].idxmin()]
    print(f"Best Origin: {best_point[:-2].to_dict()} with Stress = {best_point['stress']:.4f}")
    return pairplot
    




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def prepare_pixel_data(stress_results, dimensions):
    """
    Prepares a grid for pixel plotting based on stress results.

    Parameters:
    ----------
    stress_results : dict
        A dictionary with grid points as keys and stress values as values.

    dimensions : list
        List of dimension names (e.g., ['dim_1', 'dim_2']).

    Returns:
    -------
    pixel_data : dict
        Dictionary of 2D arrays for each pairwise combination of dimensions.
    """
    # Convert stress results into a DataFrame
    data = [list(key) + [value] for key, value in stress_results.items()]
    columns = [f"dim_{i+1}" for i in range(len(data[0]) - 1)] + ["stress"]
    stress_df = pd.DataFrame(data, columns=columns)

    pixel_data = {}
    for i, dim_x in enumerate(dimensions):
        for j, dim_y in enumerate(dimensions):
            if i >= j:  # Skip lower triangle to avoid duplicates
                continue

            # Group by dim_x and dim_y to handle duplicates
            grouped = stress_df.groupby([dim_x, dim_y], as_index=False)['stress'].mean()

            # Pivot stress values for heatmap
            grid = grouped.pivot(index=dim_y, columns=dim_x, values="stress")

            # Ensure grid is not sparse (fill missing values)
            grid = grid.fillna(grid.max().max())
            
            # Normalize the grid for visualization
            scaler = MinMaxScaler()
            normalized_grid = scaler.fit_transform(grid.values)
            
            pixel_data[(dim_x, dim_y)] = normalized_grid

    return pixel_data


def plot_pixel_matrix(pixel_data, dimensions, title="Stress Heatmap Matrix"):
    """
    Plots a matrix of pixel-based heatmaps for stress values.

    Parameters:
    ----------
    pixel_data : dict
        Dictionary of 2D arrays for each pairwise combination of dimensions.

    dimensions : list
        List of dimension names (e.g., ['dim_1', 'dim_2']).

    title : str
        Title of the overall plot.
    """
    num_dims = len(dimensions)
    fig, axes = plt.subplots(num_dims, num_dims, figsize=(10, 10))

    for i, dim_x in enumerate(dimensions):
        for j, dim_y in enumerate(dimensions):
            ax = axes[i, j]

            if i >= j:
                # Leave lower triangle blank
                ax.axis('off')
            else:
                # Plot the heatmap
                grid = pixel_data[(dim_x, dim_y)]
                im = ax.imshow(grid, cmap="viridis", origin="lower")
                ax.set_title(f"{dim_x} vs {dim_y}", fontsize=8)
                ax.axis("off")

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, location="right", shrink=0.75)
    cbar.set_label("Normalized Stress")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# Example Usage
# if __name__ == "__main__":
#     # Simulate stress results for demonstration
#     stress_results = {
#         (0.1, 0.2, 0.3, 0.4): 1.2,
#         (0.5, 0.6, 0.7, 0.8): 0.8,
#         (0.2, 0.3, 0.4, 0.5): 0.9,
#         (0.3, 0.4, 0.5, 0.6): 1.0,
#     }

#     # Dimensions to consider
#     dimensions = [f"dim_{i+1}" for i in range(4)]

#     # Prepare pixel data
#     pixel_data = prepare_pixel_data(stress_results, dimensions)

#     # Plot pixel-based heatmap matrix
#     plot_pixel_matrix(pixel_data, dimensions)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import numpy as np

    # Load Iris dataset
    iris_data = load_iris()
    points = iris_data['data']

    # Create a grid of points in the high-dimensional space with resolution 2
    grid_points = create_grid(points, resolution=10)

    # Evaluate cPro for each grid point and calculate stress
    stress_results = evaluate_cpro_on_grid(points, grid_points, lr=0.1, maxiter=100)

    # Print stress results for each grid point
    for grid_point, stress in stress_results.items():
        print(f"Grid point {grid_point}: Stress = {stress:.4f}")

    # Define dimensions for plotting
    dimensions = [f"dim_{i+1}" for i in range(points.shape[1])]

    # Prepare pixel data for heatmap matrix
    pixel_data = prepare_pixel_data(stress_results, dimensions)

    # Plot pixel-based heatmap matrix
    plot_pixel_matrix(pixel_data, dimensions)
