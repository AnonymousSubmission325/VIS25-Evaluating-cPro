import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix
from scipy.optimize import dual_annealing
import time

class CircularProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix, dual_annealing_res):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix
        self.dual_annealing_res = dual_annealing_res

    def __repr__(self):
        return f'<CircularProjectionResult 1-d-embedding={self.embedding}>'

def circular_projection(points, maxiter=10000, max_time=None, **kwargs):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using dual annealing.

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    maxiter : int, optional (default=50)
        The maximum number of iterations for the dual annealing optimization process.

    max_time : float, optional (default=None)
        The maximum allowed time in seconds for the optimization process. If None, no time limit is imposed.

    **kwargs : dict
        Additional keyword arguments passed to the dual annealing optimizer.

    Returns:
    -------
    CircularProjectionResult : object
        A result object containing:
        - `embedding` : The 1D embedding values for each point, representing their angles on the circle.
        - `circle_x` : The x-coordinates of the points projected onto the unit circle.
        - `circle_y` : The y-coordinates of the points projected onto the unit circle.
        - `loss_records` : A list containing the development of the loss values during the optimization process.
        - `hd_dist_matrix` : The high-dimensional cosine distance matrix.
        - `ld_dist_matrix` : The low-dimensional distance matrix between points on the circle.
        - `dual_annealing_res` : The result object from the dual annealing process.
    """

    # Convert input points to a numpy array if they aren't already
    points = np.array(points)
    n = points.shape[0]

    # Calculate the mean for each axis to center the points
    mean = np.mean(points, axis=0)
    points = points - mean

    # Compute high-dimensional distances using cosine distance
    hd_dist_mat = cosine_distances(points)
    hd_dist_mat = hd_dist_mat / 2  # Normalize to the range [0, 1]

    def compute_ld_dist_matrix(ld_points):
        """
        Compute the distances between points in the low-dimensional space.
        """
        dist_matrix = distance_matrix(ld_points, ld_points, p=1)
        return np.minimum(dist_matrix, 1 - dist_matrix)

    def loss(ld_points):
        """
        Compute the difference between low-dimensional and high-dimensional distances.
        """
        x = ld_points.reshape((n, 1))
        ld_dist_mat = compute_ld_dist_matrix(x)
        diff = np.absolute(hd_dist_mat - (2 * ld_dist_mat))
        return diff.sum() / 2

    # Set bounds for the low-dimensional embedding in the range (0, 1)
    bounds = [(0, 1) for _ in range(n)]
    loss_records = []

    def record_loss(_, current_loss, __):
        """
        Callback function to record the loss during optimization.
        """
        loss_records.append(current_loss)

    # Start the timer for the time constraint
    start_time = time.time()

    def time_limited_loss(x):
        """
        Loss function that stops optimization if the time limit is exceeded.
        """
        if max_time and (time.time() - start_time) > max_time:
            raise TimeoutError("Time limit exceeded during optimization.")
        return loss(x)

    # Perform dual annealing with time-limited loss function
    try:
        res = dual_annealing(time_limited_loss, bounds=bounds, callback=record_loss, maxiter=maxiter, **kwargs)
    except TimeoutError:
        print("Optimization stopped due to time limit.")
        res = None

    # Compute the circle coordinates from the final embedding
    final_embedding = res.x if res else np.zeros(n)
    circle_x = np.cos(final_embedding * 2 * np.pi)
    circle_y = np.sin(final_embedding * 2 * np.pi)

    return CircularProjectionResult(
        embedding=final_embedding,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        hd_dist_matrix=hd_dist_mat,
        ld_dist_matrix=compute_ld_dist_matrix(final_embedding.reshape(-1, 1)) if res else None,
        dual_annealing_res=res
    )
