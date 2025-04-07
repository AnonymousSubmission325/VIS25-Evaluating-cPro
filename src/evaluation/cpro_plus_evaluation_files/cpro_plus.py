import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import torch
from scipy.optimize import dual_annealing


def run_cpro(points, max_iterations=100):
    """
    Runs the cPro optimizer (Simulated Annealing-based circular projection).

    Parameters:
    ----------
    points : array-like
        The high-dimensional input data points to be projected.

    max_iterations : int, optional
        The maximum number of iterations for the optimization process.

    Returns:
    -------
    loss_records : list
        The loss values recorded during the optimization process.
    """
    points = np.array(points)
    n = points.shape[0]

    # Compute high-dimensional cosine distances
    def compute_hd_distances(points):
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / norm
        cosine_distances = 1 - np.dot(normalized_points, normalized_points.T)
        return cosine_distances / 2  # Normalize to [0, 1]

    hd_distances = compute_hd_distances(points)

    # Low-dimensional distance matrix calculation
    def compute_ld_distances(ld_points):
        ld_distances = np.abs(ld_points[:, None] - ld_points[None, :])
        return np.minimum(ld_distances, 1 - ld_distances)

    # Loss function
    def loss(ld_points):
        ld_distances = compute_ld_distances(ld_points)
        return np.sum(np.abs(hd_distances - (2 * ld_distances))) / 2

    # Bounds for optimization
    bounds = [(0, 1) for _ in range(n)]
    loss_records = []

    # Callback to record loss
    def record_loss(x, f, context):
        loss_records.append(f)

    # Run simulated annealing
    dual_annealing(loss, bounds, callback=record_loss, maxiter=max_iterations)

    return loss_records


def run_cpro_plus(points, lr=0.1, max_iterations=100):
    """
    Runs the cPro+ optimizer (Adam-based circular projection).

    Parameters:
    ----------
    points : array-like
        The high-dimensional input data points to be projected.

    lr : float, optional
        The learning rate for the Adam optimizer.

    max_iterations : int, optional
        The maximum number of iterations for the optimization process.

    Returns:
    -------
    loss_records : list
        The loss values recorded during the optimization process.
    """
    # Convert points to torch tensors
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]

    # Center the points
    points = points - torch.mean(points, dim=0)

    # Compute high-dimensional distances using cosine distance
    def compute_hd_distances(points):
        norm = torch.norm(points, dim=1, keepdim=True)
        normalized_points = points / norm
        cosine_distances = 1 - torch.mm(normalized_points, normalized_points.T)
        return cosine_distances

    hd_distances = compute_hd_distances(points)

    # Initialize low-dimensional embedding
    embedding = torch.randn(n, requires_grad=True)

    # Adam optimizer
    optimizer = torch.optim.Adam([embedding], lr=lr)

    # Loss function
    def compute_ld_distances(embedding):
        embedding = embedding.view(-1, 1)
        ld_distances = torch.cdist(embedding, embedding, p=1)
        return torch.minimum(ld_distances, 1 - ld_distances)

    def loss_function(embedding):
        ld_distances = compute_ld_distances(embedding)
        loss = torch.sum(torch.abs(hd_distances - 2 * ld_distances)) / 2
        return loss

    # Optimization loop
    loss_records = []
    for _ in range(max_iterations):
        optimizer.zero_grad()
        loss = loss_function(embedding)
        loss.backward()
        optimizer.step()
        loss_records.append(loss.item())

    return loss_records


def load_sample_data():
    """
    Loads the Iris dataset for testing.
    Returns the dataset's feature matrix.
    """
    iris_data = load_iris()
    return {"Iris Dataset": iris_data["data"]}


def plot_comparison(dataset_name, loss_records_cpro, loss_records_cpro_plus, save_dir="results"):
    """
    Plots the loss values for cPro and cPro+ optimizers.

    Parameters:
    ----------
    dataset_name : str
        Name of the dataset.

    loss_records_cpro : list
        The loss values recorded during the cPro optimization process.

    loss_records_cpro_plus : list
        The loss values recorded during the cPro+ optimization process.

    save_dir : str, optional
        Directory to save the plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))

    plt.plot(loss_records_cpro, label="cPro Loss (Simulated Annealing)", linewidth=2, color="red")
    plt.plot(loss_records_cpro_plus, label="cPro+ Loss (Adam)", linewidth=2, color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve Comparison - {dataset_name}")
    plt.legend()

    save_path = os.path.join(save_dir, f"{dataset_name.replace(' ', '_')}_cpro_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    datasets = load_sample_data()
    for dataset_name, points in datasets.items():
        print(f"Running cPro and cPro+ on {dataset_name} dataset...")

        print("Running cPro (Simulated Annealing)...")
        loss_records_cpro = run_cpro(points, max_iterations=100)

        print("Running cPro+ (Adam)...")
        loss_records_cpro_plus = run_cpro_plus(points, lr=0.1, max_iterations=100)

        plot_comparison(dataset_name, loss_records_cpro, loss_records_cpro_plus)
        print(f"Finished processing {dataset_name} dataset.")
