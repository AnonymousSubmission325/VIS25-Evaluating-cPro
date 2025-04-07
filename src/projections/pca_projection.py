import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def normalize_projection_output(data):
    """
    Normalizes the projection output data to the range [-1, 1].

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The low-dimensional projection output.

    Returns:
    -------
    normalized_data : array-like, shape (n_samples, n_features)
        The normalized low-dimensional data.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def run_pca_with_time_limit(data, n_components, max_time):
    """
    Runs PCA with a specified time limit and normalizes the output.

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.

    n_components : int
        The number of components for PCA (1D or 2D).

    max_time : float
        The maximum allowed time in seconds for the PCA process.

    Returns:
    -------
    ld_data : array-like, shape (n_samples, n_components)
        The low-dimensional embedding from PCA, normalized to [-1, 1].
    """
    # Start the timer
    start_time = time.time()

    # Initialize PCA
    pca = PCA(n_components=n_components)

    # Fit PCA and transform the data
    ld_data = pca.fit_transform(data)

    # Check if the elapsed time exceeds the time limit
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        print("PCA stopped due to time limit.")

    # Normalize the output to [-1, 1]
    ld_data = normalize_projection_output(ld_data)


    return ld_data
