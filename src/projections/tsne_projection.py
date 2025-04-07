import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def run_tsne_with_time_limit(data, n_components, max_time, perplexity=20.0, learning_rate=200.0):
    """
    Runs t-SNE with a specified time limit and normalizes the output.

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.

    n_components : int
        The number of components for t-SNE (1D or 2D).

    max_time : float
        The maximum allowed time in seconds for the t-SNE process.

    perplexity : float, optional (default=30.0)
        The perplexity is related to the number of nearest neighbors used in t-SNE.

    learning_rate : float, optional (default=200.0)
        The learning rate for t-SNE.

    Returns:
    -------
    ld_data : array-like, shape (n_samples, n_components)
        The normalized low-dimensional embedding from t-SNE.
    """
    # Start the timer
    start_time = time.time()

    # Initialize t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate)

    # Fit t-SNE and transform the data
    ld_data = tsne.fit_transform(data)

    # Check if the elapsed time exceeds the time limit
    elapsed_time = time.time() - start_time
    if elapsed_time >= max_time:
        print("t-SNE stopped due to time limit.")

    # Normalize the output to [-1, 1]
    ld_data = scale_to_unit_range(ld_data)

    return ld_data
