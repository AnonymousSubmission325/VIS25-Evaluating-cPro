import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix
from pyswarm import pso  # PSO implementation
import time

class PSOCircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix, pso_res):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix
        self.pso_res = pso_res

    def __repr__(self):
        return f'<PSOCircularProjectionResult 1-d-embedding={self.embedding}>'

def circular_projection_pso(points, maxiter=6000, swarmsize=3000, max_time=None, **kwargs):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using PSO.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    maxiter : int, optional (default=6000)
        The maximum number of iterations for the PSO optimization process.

    swarmsize : int, optional (default=4000)
        The number of particles in the swarm (i.e., candidate solutions).

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    **kwargs : dict
        Additional keyword arguments passed to the PSO optimizer.

    Returns:
    -------
    PSOCircularProjectionResult : object
        A result object containing the projection and metadata.
    """

    # Make sure we are dealing with a numpy array
    points = np.array(points)

    # Number of data points
    n = points.shape[0]

    # Calculate the mean for each axis and center the points
    mean = np.mean(points, axis=0)
    points = points - mean

    # Calculate distances in the high-dimensional space using cosine distance
    hd_dist_mat = cosine_distances(points)
    hd_dist_mat = hd_dist_mat / 2  # Normalize to [0, 1]

    # Store the loss records for tracking
    loss_records = []

    def compute_ld_dist_matrix(ld_points):
        """Compute the distances between points in the low-dimensional space."""
        dist_matrix = distance_matrix(ld_points, ld_points, p=1)
        return np.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        """Compute the difference between low-dimensional and high-dimensional distances."""
        x = ld_points.reshape((n, 1))
        ld_dist_mat = compute_ld_dist_matrix(x)
        diff = np.absolute(hd_dist_mat - (2 * ld_dist_mat))
        loss_value = diff.sum() / 2
        loss_records.append(loss_value)
        return loss_value

    # Lower and upper bounds for the 1D embedding
    lb = np.zeros(n)
    ub = np.ones(n)

    # Start the timer for the time constraint
    start_time = time.time()

    def timed_pso_loss(ld_points):
        """A loss function that also checks the time constraint."""
        if max_time and (time.time() - start_time) > max_time:
            raise TimeoutError("Optimization stopped due to time limit.")
        return loss(ld_points)

    try:
        # Perform PSO to minimize the loss function
        pso_res, pso_loss = pso(timed_pso_loss, lb, ub, maxiter=maxiter, swarmsize=swarmsize, **kwargs)
    except TimeoutError as e:
        print(e)
        pso_res = np.zeros(n)  # Return a default result if time limit is exceeded
        pso_loss = float('inf')  # Assign a high loss value

    # Calculate the circle coordinates from the low-dimensional embedding
    circle_x = np.cos(pso_res * 2 * np.pi)
    circle_y = np.sin(pso_res * 2 * np.pi)

    return PSOCircularProjectionResult(
        embedding=pso_res,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        hd_dist_matrix=hd_dist_mat,
        ld_dist_matrix=compute_ld_dist_matrix(pso_res.reshape(-1, 1)),
        pso_res=pso_res
    )

# Example usage:
# points = np.random.rand(100, 5)
# result = circular_projection_pso(points, max_time=60)  # Limit the optimization to 60 seconds
# print(result.circle_x, result.circle_y)


