import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import distance_matrix

class ALSProjectionResult:
    def __init__(self, embedding, circle_x, circle_y, loss_records, hd_dist_matrix, ld_dist_matrix):
        self.embedding = embedding
        self.circle_x = circle_x
        self.circle_y = circle_y
        self.loss_records = loss_records
        self.hd_dist_matrix = hd_dist_matrix
        self.ld_dist_matrix = ld_dist_matrix

    def __repr__(self):
        return f'<ALSProjectionResult 1-d-embedding={self.embedding}>'


def circular_projection_als(points, maxiter=100):
    """
    Projects high-dimensional data points onto a circle while preserving relative distances using Alternating Least Squares (ALS).

    Parameters:
    ----------
    points : array-like, shape (n_samples, n_features)
        The high-dimensional input data points to be projected.

    maxiter : int, optional (default=100)
        The maximum number of iterations for the ALS optimization process.

    Returns:
    -------
    ALSProjectionResult : object
        A result object containing:
        - `embedding` : The 1D embedding values for each point, representing their angles on the circle.
        - `circle_x` : The x-coordinates of the points projected onto the unit circle.
        - `circle_y` : The y-coordinates of the points projected onto the unit circle.
        - `loss_records` : A list containing the development of the loss values during the optimization process.
        - `hd_dist_matrix` : The high-dimensional cosine distance matrix.
        - `ld_dist_matrix` : The low-dimensional distance matrix between points on the circle.
    """
    
    # Convert points to numpy array
    points = np.array(points)
    n = points.shape[0]
    
    # Center the points
    mean = np.mean(points, axis=0)
    points = points - mean

    # Calculate distances in the high-dimensional space using cosine-distance
    hd_dist_mat = cosine_distances(points)
    hd_dist_mat = hd_dist_mat / 2  # Normalize cosine distance to [0, 1]

    # Initialize embedding randomly as angles on a circle
    embedding = np.random.rand(n)

    # Loss records for tracking progress
    loss_records = []

    def compute_ld_dist_matrix(ld_points):
        """Compute the pairwise distances in low-dimensional (1D) space."""
        ld_points = ld_points.reshape((n, 1))
        dist_matrix = distance_matrix(ld_points, ld_points, p=1)
        return np.minimum(dist_matrix, 1 - dist_matrix)

    def loss_function(ld_points):
        """Compute the difference between the high-dimensional and low-dimensional distances."""
        ld_dist_mat = compute_ld_dist_matrix(ld_points)
        return np.sum(np.abs(hd_dist_mat - (2 * ld_dist_mat)))

    # ALS Optimization process
    for iteration in range(maxiter):
        # Fix embedding and update distances iteratively
        ld_dist_mat = compute_ld_dist_matrix(embedding)

        # ALS Optimization: Update the embedding by minimizing the distance differences
        for i in range(n):
            other_distances = np.delete(ld_dist_mat[i], i)
            embedding[i] = np.mean(other_distances)

        # Compute current loss and track it
        current_loss = loss_function(embedding)
        loss_records.append(current_loss)

    # Calculate the final positions on the circle
    circle_x = np.cos(embedding * 2 * np.pi)
    circle_y = np.sin(embedding * 2 * np.pi)

    return ALSProjectionResult(
        embedding=embedding,
        circle_x=circle_x,
        circle_y=circle_y,
        loss_records=loss_records,
        hd_dist_matrix=hd_dist_mat,
        ld_dist_matrix=compute_ld_dist_matrix(embedding)
    )

# Example usage:
# points = np.random.rand(100, 5)
# result = circular_projection_als(points)
# print(result.circle_x, result.circle_y)
