import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, euclidean_distances, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, manhattan_distances
from scipy.stats import pearsonr
from src.projections.circular_projection import CircularProjectionResult


def calculate_stress(hd_distances, ld_distances):
    return np.sqrt(np.sum((hd_distances - ld_distances) ** 2) / np.sum(hd_distances ** 2))


def calculate_correlation(hd_distances, ld_distances):
    return pearsonr(hd_distances.flatten(), ld_distances.flatten())[0]


def calculate_trustworthiness(x_high, x_low, n_neighbors=5, distance='euclidean'):
    n = x_high.shape[0]

    # Compute nearest neighbors in the original high-dimensional space
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance).fit(x_high)
    _, indices_orig = nn_orig.kneighbors(x_high)

    # Compute nearest neighbors in the low-dimensional space
    nn_proj = NearestNeighbors(n_neighbors=n, metric=distance).fit(x_low)
    _, indices_proj = nn_proj.kneighbors(x_low)

    # Calculate the rank matrix
    rank_matrix = np.full((n, n_neighbors), n)
    for i in range(n):
        for j in range(1, n_neighbors + 1):
            high_neighbor = indices_orig[i, j]
            if high_neighbor in indices_proj[i]:
                low_neighbor_rank = np.where(indices_proj[i] == high_neighbor)[0][0]
                rank_matrix[i, j - 1] = low_neighbor_rank

    # Subtract (n_neighbors + 1) from each element in the rank matrix
    rank_matrix -= (n_neighbors + 1)

    # Calculate the trustworthiness
    trustworthiness = 1 - (2.0 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1)) *
                           np.sum(rank_matrix[rank_matrix > 0]))
    return trustworthiness


def calculate_average_distance(x, distance='euclidean'):
    """
    Calculates Euclidean, Cosine, or Manhattan distances.
    Averages all distances in a matrix as a proxy for compactness.
    """
    if distance == 'cosine':
        dist = cosine_distances(x)
    elif distance == 'manhattan':
        dist = manhattan_distances(x)
    elif distance == 'euclidean':
        dist = euclidean_distances(x)
    else:
        raise 'Unknown distance type'
    return np.mean(dist)


def run_evaluation(hd_data, labels, circular_proj_res: CircularProjectionResult, ld_data_other):
    """
    Evaluates and compares the performance of the circular projection technique against another
    low-dimensional embedding method using various distance metrics and quality measures.

    Computes a variety of distance-based metrics (Euclidean, Cosine, Manhattan) and evaluates the quality
    of both the circular projection and another low-dimensional embedding technique. Calculates stress, correlation,
    silhouette scores, trustworthiness, and average distances for each method, and prints the results in a formatted table.

    Parameters:
    -----------
    hd_data : array-like, shape (n_samples, n_features)
        The original data points in the high-dimensional space.

    labels : array-like, shape (n_samples,)
        The cluster labels or class labels associated with the high-dimensional data points, used for silhouette score evaluation.

    circular_proj_res : CircularProjectionResult
        The result of the circular projection, containing the projected x and y coordinates of the points.

    ld_data_other : array-like, shape (n_samples, 2)
        The low-dimensional embeddings of the data points produced by another projection technique, for comparison with the circular projection.

    Returns:
    --------
    None
        The function does not return any value. It prints out a table comparing the circular projection with the other low-dimensional
        embedding method across multiple metrics (stress, correlation, silhouette score, trustworthiness, and average distance) for
        different distance metrics (Euclidean, Cosine, Manhattan).
    """
    # Extract the x-y-coordinates from the circular projection
    ld_data_circle = pd.DataFrame({
        'x': circular_proj_res.circle_x,
        'y': circular_proj_res.circle_y,
    })

    # Calculate average distance for each projection and distance metric
    avg_dist_euclidean_other = calculate_average_distance(ld_data_other, distance='euclidean')
    avg_dist_euclidean_circle = calculate_average_distance(ld_data_circle, distance='euclidean')

    avg_dist_cosine_other = calculate_average_distance(ld_data_other, distance='cosine')
    avg_dist_cosine_circle = calculate_average_distance(ld_data_circle, distance='cosine')

    avg_dist_manhattan_other = calculate_average_distance(ld_data_other, distance='manhattan')
    avg_dist_manhattan_circle = calculate_average_distance(ld_data_circle, distance='manhattan')

    # Euclidean distances
    hd_euclidean_dist = euclidean_distances(hd_data)
    ld_euclidean_dist_other = euclidean_distances(ld_data_other)
    ld_euclidean_dist_circle = euclidean_distances(ld_data_circle)

    # Cosine distances
    hd_cosine_dist = cosine_distances(hd_data)
    ld_cosine_dist_other = cosine_distances(ld_data_other)
    ld_cosine_dist_circle = cosine_distances(ld_data_circle)

    # Manhattan distances
    hd_manhattan_dist = manhattan_distances(hd_data)
    ld_manhattan_dist_other = manhattan_distances(ld_data_other)
    ld_manhattan_dist_circle = manhattan_distances(ld_data_circle)

    # Stress and correlation calculations for each metric
    # Euclidean
    stress_euclidean_other = calculate_stress(hd_euclidean_dist, ld_euclidean_dist_other)
    stress_euclidean_circle = calculate_stress(hd_euclidean_dist, ld_euclidean_dist_circle)
    correlation_euclidean_other = calculate_correlation(hd_euclidean_dist, ld_euclidean_dist_other)
    correlation_euclidean_circle = calculate_correlation(hd_euclidean_dist, ld_euclidean_dist_circle)

    # Cosine
    stress_cosine_other = calculate_stress(hd_cosine_dist, ld_cosine_dist_other)
    stress_cosine_circle = calculate_stress(hd_cosine_dist, ld_cosine_dist_circle)
    correlation_cosine_other = calculate_correlation(hd_cosine_dist, ld_cosine_dist_other)
    correlation_cosine_circle = calculate_correlation(hd_cosine_dist, ld_cosine_dist_circle)

    # Manhattan
    stress_manhattan_other = calculate_stress(hd_manhattan_dist, ld_manhattan_dist_other)
    stress_manhattan_circle = calculate_stress(hd_manhattan_dist, ld_manhattan_dist_circle)
    correlation_manhattan_other = calculate_correlation(hd_manhattan_dist, ld_manhattan_dist_other)
    correlation_manhattan_circle = calculate_correlation(hd_manhattan_dist, ld_manhattan_dist_circle)

    # Silhouette scores for each metric
    silhouette_euclidean_other = silhouette_score(ld_data_other, labels)
    silhouette_euclidean_circle = silhouette_score(ld_data_circle, labels)

    silhouette_cosine_other = silhouette_score(ld_data_other, labels, metric='cosine')
    silhouette_cosine_circle = silhouette_score(ld_data_circle, labels, metric='cosine')

    silhouette_manhattan_other = silhouette_score(ld_data_other, labels, metric='manhattan')
    silhouette_manhattan_circle = silhouette_score(ld_data_circle, labels, metric='manhattan')

    # Euclidean
    trust_euclidean_other = calculate_trustworthiness(hd_data, ld_data_other, distance='euclidean')
    trust_euclidean_circle = calculate_trustworthiness(hd_data, ld_data_circle, distance='euclidean')

    # Cosine
    trust_cosine_other = calculate_trustworthiness(hd_data, ld_data_other, distance='cosine')
    trust_cosine_circle = calculate_trustworthiness(hd_data, ld_data_circle, distance='cosine')

    # Manhattan
    trust_manhattan_other = calculate_trustworthiness(hd_data, ld_data_other, distance='manhattan')
    trust_manhattan_circle = calculate_trustworthiness(hd_data, ld_data_circle, distance='manhattan')

    # Print results as a table
    print(f"|                         | Circular Projection | Other Projection Method |")
    print(f"| ----------------------------------------------------------------------- |")
    print(f"| Euclidean - Stress      | {stress_euclidean_circle:19.3f} |     {stress_euclidean_other:19.3f} |")
    print(f"| Euclidean - Correlation | {correlation_euclidean_circle:19.3f} |     {correlation_euclidean_other:19.3f} |")
    print(f"| Euclidean - Silhouette  | {silhouette_euclidean_circle:19.3f} |     {silhouette_euclidean_other:19.3f} |")
    print(f"| Euclidean - Trustworth. | {trust_euclidean_circle:19.3f} |     {trust_euclidean_other:19.3f} |")
    print(f"| Euclidean - Avg. Dist.  | {avg_dist_euclidean_circle:19.3f} |     {avg_dist_euclidean_other:19.3f} |")
    print(f"| ----------------------------------------------------------------------- |")
    print(f"| Cosine - Stress         | {stress_cosine_circle:19.3f} |     {stress_cosine_other:19.3f} |")
    print(f"| Cosine - Correlation    | {correlation_cosine_circle:19.3f} |     {correlation_cosine_other:19.3f} |")
    print(f"| Cosine - Silhouette     | {silhouette_cosine_circle:19.3f} |     {silhouette_cosine_other:19.3f} |")
    print(f"| Cosine - Trustworth.    | {trust_cosine_circle:19.3f} |     {trust_cosine_other:19.3f} |")
    print(f"| Cosine - Avg. Dist.     | {avg_dist_cosine_circle:19.3f} |     {avg_dist_cosine_other:19.3f} |")
    print(f"| ----------------------------------------------------------------------- |")
    print(f"| Manhattan - Stress      | {stress_manhattan_circle:19.3f} |     {stress_manhattan_other:19.3f} |")
    print(f"| Manhattan - Correlation | {correlation_manhattan_circle:19.3f} |     {correlation_manhattan_other:19.3f} |")
    print(f"| Manhattan - Silhouette  | {silhouette_manhattan_circle:19.3f} |     {silhouette_manhattan_other:19.3f} |")
    print(f"| Manhattan - Trustworth. | {trust_manhattan_circle:19.3f} |     {trust_manhattan_other:19.3f} |")
    print(f"| Manhattan - Avg. Dist.  | {avg_dist_manhattan_circle:19.3f} |     {avg_dist_manhattan_other:19.3f} |")
    print(f"| ----------------------------------------------------------------------- |")

