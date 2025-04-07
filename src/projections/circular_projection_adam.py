import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix
import pandas as pd
import time

class AdamCircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<AdamCircularProjectionResult 1-d-embedding={self.embedding}>'

def circular_projection_adam(points, lr=0.1, maxiter=20, max_time=None):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using Adam.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    lr : float, optional (default=0.1)
        The learning rate for the Adam optimizer.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    Returns:
    -------
    AdamCircularProjectionResult : object
        A result object containing the projection and metadata.
    """

    # Convert points to a numpy array if it's a DataFrame
    if isinstance(points, pd.DataFrame):
        points = points.values

    # Convert points to torch tensors for optimization
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]
    
    # Calculate the mean for each axis and center the points
    mean = torch.mean(points, dim=0)
    points = points - mean

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

    return AdamCircularProjectionResult(
        embedding=final_embedding,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        hd_dist_matrix=hd_dist_mat.detach().numpy(),
        ld_dist_matrix=compute_ld_dist_matrix(embedding).detach().numpy()
    )









# In loss_wrappers.py

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import pandas as pd
import time

def run_adam_cpro_loss(points, lr=0.1, maxiter=100, max_time=None):
    """
    Runs the circular projection using Adam optimizer and returns the loss records.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    lr : float, optional (default=0.1)
        The learning rate for the Adam optimizer.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    Returns:
    -------
    loss_records : list
        A list containing the loss values during the optimization process.
    """

    # Convert points to a numpy array if they are provided as a DataFrame
    if isinstance(points, pd.DataFrame):
        points = points.values

    # Convert points to torch tensors for optimization
    points = torch.tensor(points, dtype=torch.float32)
    n = points.shape[0]
    
    # Center the points
    points = points - torch.mean(points, dim=0)

    # Calculate distances in the high-dimensional space using cosine distance
    hd_dist_mat = cosine_distances(points.detach().numpy()) / 2
    hd_dist_mat = torch.tensor(hd_dist_mat, dtype=torch.float32)

    # Initialize the embedding as learnable parameters
    embedding = torch.randn(n, requires_grad=True)

    # Initialize the Adam optimizer
    optimizer = torch.optim.Adam([embedding], lr=lr)

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

    # Start timer for time constraint
    start_time = time.time()

    # Optimization loop
    for i in range(maxiter):
        if max_time and (time.time() - start_time) > max_time:
            print("Optimization stopped due to time limit.")
            break

        optimizer.zero_grad()
        current_loss = loss(embedding)
        current_loss.backward()
        optimizer.step()

        # Record loss
        loss_records.append(current_loss.item())

    return loss_records
