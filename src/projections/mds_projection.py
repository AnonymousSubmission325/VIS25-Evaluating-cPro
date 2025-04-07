import time
import numpy as np
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def run_mds_with_time_limit(data, n_components, max_time, random_state=777):
    """
    Runs MDS with a specified time limit and normalizes the output.

    Parameters:
    ----------
    data : array-like, shape (n_samples, n_features)
        The input high-dimensional data.

    n_components : int
        The number of components for MDS (1D or 2D).

    max_time : float
        The maximum allowed time in seconds for the MDS process.

    random_state : int, optional
        The random seed for reproducibility.

    Returns:
    -------
    ld_data : array-like, shape (n_samples, n_components)
        The normalized low-dimensional embedding from MDS.
    """
    # Initialize MDS with a single iteration to check progress
    mds = MDS(n_components=n_components, random_state=random_state, max_iter=1, normalized_stress="auto")

    # Start the timer
    start_time = time.time()
    elapsed_time = 0
    prev_embedding = None

    # Iteratively run MDS until the time limit is reached
    while elapsed_time < max_time:
        # Fit the MDS model and transform the data
        embedding = mds.fit_transform(data)

        # Check if the embedding has stabilized (optional)
        if prev_embedding is not None and np.allclose(embedding, prev_embedding, atol=1e-5):
            print("MDS embedding stabilized.")
            break

        prev_embedding = embedding
        elapsed_time = time.time() - start_time

    if elapsed_time >= max_time:
        print("MDS stopped due to time limit.")

    # Normalize the output to [-1, 1]
    embedding = scale_to_unit_range(embedding)

    return embedding
