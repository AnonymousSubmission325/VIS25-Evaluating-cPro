import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import time

class LbfgsCircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<LbfgsCircularProjectionResult 1-d-embedding={self.embedding}>'

def circular_projection_lbfgs(points, maxiter=100, max_time=None):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using L-BFGS,
    with a time constraint.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    Returns:
    -------
    LbfgsCircularProjectionResult : object
        A result object containing the embedding and projection coordinates.
    """

    # Convert points to numpy array if it's a DataFrame, then to a torch tensor
    if isinstance(points, pd.DataFrame):
        points = points.values
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]

    # Calculate distances in the high-dimensional space using cosine distance
    hd_dist_mat = cosine_distances(points.detach().numpy())
    hd_dist_mat = torch.tensor(hd_dist_mat / 2, dtype=torch.float32)

    # Initialize the embedding (1D positions on the circle) as learnable parameters
    embedding = torch.randn(n, requires_grad=True)

    # Record the loss during iterations
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

    # L-BFGS optimizer from PyTorch
    optimizer = torch.optim.LBFGS([embedding], max_iter=maxiter, line_search_fn='strong_wolfe')

    # Start the timer for the time constraint
    start_time = time.time()

    # Optimization loop using L-BFGS
    def closure():
        optimizer.zero_grad()
        total_loss = loss(embedding)
        total_loss.backward()
        return total_loss

    for i in range(maxiter):
        # Check the time constraint
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break

        optimizer.step(closure)
        loss_value = closure().item()
        loss_records.append(loss_value)

    # Convert the embedding to angles on the circle
    final_embedding = embedding.detach().numpy()
    circle_x = np.cos(final_embedding * 2 * np.pi)
    circle_y = np.sin(final_embedding * 2 * np.pi)

    return LbfgsCircularProjectionResult(
        embedding=final_embedding,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        hd_dist_matrix=hd_dist_mat.detach().numpy(),
        ld_dist_matrix=compute_ld_dist_matrix(embedding).detach().numpy()
    )

# Example usage:
# points = np.random.rand(100, 5)
# result = circular_projection_lbfgs(points, max_time=30)  # Limit the optimization to 30 seconds
# print(result.circle_x, result.circle_y)


# In loss_wrappers.py

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import time

def run_lbfgs_cpro_loss(points, maxiter=100, max_time=None):
    """
    Runs the circular projection using L-BFGS optimizer and returns the loss records.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    Returns:
    -------
    loss_records : list
        A list containing the loss values during the optimization process.
    """

    # Convert points to numpy array if they are provided as a DataFrame, then to a torch tensor
    if isinstance(points, pd.DataFrame):
        points = points.values
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]

    # Calculate distances in the high-dimensional space using cosine distance
    hd_dist_mat = cosine_distances(points.detach().numpy()) / 2
    hd_dist_mat = torch.tensor(hd_dist_mat, dtype=torch.float32)

    # Initialize the embedding as learnable parameters
    embedding = torch.randn(n, requires_grad=True)

    # Loss records list
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

    # L-BFGS optimizer from PyTorch
    optimizer = torch.optim.LBFGS([embedding], max_iter=maxiter, line_search_fn='strong_wolfe')

    # Start timer for the time constraint
    start_time = time.time()

    def closure():
        optimizer.zero_grad()
        total_loss = loss(embedding)
        total_loss.backward()
        return total_loss

    # Optimization loop with time check
    for i in range(maxiter):
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break

        optimizer.step(closure)
        current_loss = closure().item()
        loss_records.append(current_loss)

    return loss_records
