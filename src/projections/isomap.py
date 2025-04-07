import time
import numpy as np
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def run_isomap_with_time_limit(data, n_components, max_time, n_neighbors=15):
    """
    Runs Isomap with a specified time limit and normalizes the output.

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.

    n_components : int
        The number of components for Isomap (1D or 2D).

    max_time : float
        The maximum allowed time in seconds for the Isomap process.

    n_neighbors : int, optional (default=5)
        The number of neighbors to consider for each point.

    Returns:
    -------
    ld_data : array-like, shape (n_samples, n_components)
        The normalized low-dimensional embedding from Isomap.
    """
    # Start the timer
    start_time = time.time()

    # Initialize Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)

    # Fit Isomap and transform the data
    ld_data = isomap.fit_transform(data)

    # Check if the elapsed time exceeds the time limit
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        print("Isomap stopped due to time limit.")

    # Normalize the output to [-1, 1]
    ld_data = scale_to_unit_range(ld_data)

    return ld_data
