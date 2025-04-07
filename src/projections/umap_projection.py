import time
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def run_umap_with_time_limit(data, n_components, max_time, n_neighbors=15, min_dist=0.1):
    """
    Runs UMAP with a specified time limit and normalizes the output.

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.

    n_components : int
        The number of components for UMAP (1D or 2D).

    max_time : float
        The maximum allowed time in seconds for the UMAP process.

    n_neighbors : int, optional (default=15)
        The number of neighbors to consider for manifold approximation.

    min_dist : float, optional (default=0.1)
        The minimum distance between points in the low-dimensional space.

    Returns:
    -------
    ld_data : array-like, shape (n_samples, n_components)
        The normalized low-dimensional embedding from UMAP.
    """
    # Start the timer
    start_time = time.time()

    # Initialize UMAP
    umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)

    # Fit UMAP and transform the data
    ld_data = umap_model.fit_transform(data)

    # Check if the elapsed time exceeds the time limit
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        print("UMAP stopped due to time limit.")

    # Normalize the output to [-1, 1]
    ld_data = scale_to_unit_range(ld_data)

    return ld_data
