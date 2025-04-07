import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
from src.projections.som_projection import SOMProjectionResult
from src.projections.circular_projection import CircularProjectionResult
from src.projections.circular_projection_adam import AdamCircularProjectionResult
from src.projections.circular_projection_pso import PSOCircularProjectionResult
from src.projections.circular_projection_lbfgs import LbfgsCircularProjectionResult
from src.projections.tow import SpringForceProjectionResult

def calculate_stress(hd_distances, ld_distances):
    return np.sqrt(np.sum((hd_distances - ld_distances) ** 2) / np.sum(hd_distances ** 2))

def calculate_correlation(hd_distances, ld_distances):
    return pearsonr(hd_distances.flatten(), ld_distances.flatten())[0]

def calculate_trustworthiness(x_high, x_low, n_neighbors=5, distance='cosine'):
    n = x_high.shape[0]

    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1, metric=distance).fit(x_high)
    _, indices_orig = nn_orig.kneighbors(x_high)

    nn_proj = NearestNeighbors(n_neighbors=n_neighbors, metric=distance).fit(x_low)
    _, indices_proj = nn_proj.kneighbors(x_low)

    rank_matrix = np.full((n, n_neighbors), n)
    for i in range(n):
        for j in range(1, n_neighbors + 1):
            high_neighbor = indices_orig[i, j]
            if high_neighbor in indices_proj[i]:
                low_neighbor_rank = np.where(indices_proj[i] == high_neighbor)[0][0]
                rank_matrix[i, j - 1] = low_neighbor_rank

    rank_matrix -= (n_neighbors + 1)
    trustworthiness = 1 - (2.0 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1)) *
                           np.sum(rank_matrix[rank_matrix > 0]))
    return trustworthiness

def calculate_average_distance(x, distance='cosine'):
    dist = cosine_distances(x)
    return np.mean(dist)

def convert_projection_to_array(projection):
    """
    Converts various custom projection results to a 2D NumPy array.
    """
    if isinstance(projection, (CircularProjectionResult, AdamCircularProjectionResult)):
        return np.column_stack((projection.circle_x, projection.circle_y))
    elif isinstance(projection, SOMProjectionResult):
        return projection.embedding.numpy()
    elif isinstance(projection, PSOCircularProjectionResult):
        return np.column_stack((projection.circle_x, projection.circle_y))
    elif isinstance(projection, LbfgsCircularProjectionResult):
        return np.column_stack((projection.circle_x, projection.circle_y))
    elif isinstance(projection, SpringForceProjectionResult):
        return np.column_stack((projection.circle_x, projection.circle_y))
    elif isinstance(projection, np.ndarray):
        return projection
    else:
        raise TypeError(f"Unsupported projection format: {type(projection)}")


def calculate_continuity(hd_data, ld_data, n_neighbors=5):
    n = hd_data.shape[0]

    nn_hd = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(hd_data)
    _, indices_hd = nn_hd.kneighbors(hd_data)

    nn_ld = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(ld_data)
    _, indices_ld = nn_ld.kneighbors(ld_data)

    continuity = 0
    for i in range(n):
        hd_neighbors = set(indices_hd[i][1:])
        ld_neighbors = set(indices_ld[i][1:])
        continuity += len(hd_neighbors & ld_neighbors)

    continuity /= (n * n_neighbors)
    return continuity

def calculate_neighborhood_hit(hd_data, ld_data, labels, n_neighbors=5):
    n = ld_data.shape[0]

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(ld_data)
    _, indices = nn.kneighbors(ld_data)

    neighborhood_hit = 0
    for i in range(n):
        neighbor_labels = labels[indices[i][1:]]
        neighborhood_hit += np.sum(neighbor_labels == labels[i])

    neighborhood_hit /= (n * n_neighbors)
    return neighborhood_hit

def calculate_shepard_goodness(hd_distances, ld_distances):
    return pearsonr(hd_distances.flatten(), ld_distances.flatten())[0]

def calculate_distance_consistency(hd_data, ld_data, n_neighbors=5):
    n = hd_data.shape[0]

    nn_hd = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(hd_data)
    _, indices_hd = nn_hd.kneighbors(hd_data)

    nn_ld = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(ld_data)
    _, indices_ld = nn_ld.kneighbors(ld_data)

    consistency = 0
    for i in range(n):
        hd_neighbors = set(indices_hd[i][1:])
        ld_neighbors = set(indices_ld[i][1:])
        consistency += len(hd_neighbors & ld_neighbors)

    consistency /= n
    return consistency

def run_evaluation(hd_data, labels, circular_proj_res: CircularProjectionResult, other_projections):
    ld_data_circle = convert_projection_to_array(circular_proj_res)

    technique_names = ['Circular Projection']
    stress_values = []
    correlation_values = []
    silhouette_values = []
    trust_values = []
    avg_dist_values = []
    continuity_values = []
    neighborhood_hit_values = []
    shepard_goodness_values = []
    distance_consistency_values = []

    hd_cosine_dist = cosine_distances(hd_data)
    ld_cosine_dist_circle = cosine_distances(ld_data_circle)

    stress_cosine_circle = calculate_stress(hd_cosine_dist, ld_cosine_dist_circle)
    correlation_cosine_circle = calculate_correlation(hd_cosine_dist, ld_cosine_dist_circle)
    silhouette_cosine_circle = silhouette_score(ld_data_circle, labels, metric='cosine')
    trust_cosine_circle = calculate_trustworthiness(hd_data, ld_data_circle, distance='cosine')
    avg_dist_cosine_circle = calculate_average_distance(ld_data_circle, distance='cosine')
    continuity_circle = calculate_continuity(hd_data, ld_data_circle)
    neighborhood_hit_circle = calculate_neighborhood_hit(hd_data, ld_data_circle, labels)
    shepard_goodness_circle = calculate_shepard_goodness(hd_cosine_dist, ld_cosine_dist_circle)
    distance_consistency_circle = calculate_distance_consistency(hd_data, ld_data_circle)

    stress_values.append(stress_cosine_circle)
    correlation_values.append(correlation_cosine_circle)
    silhouette_values.append(silhouette_cosine_circle)
    trust_values.append(trust_cosine_circle)
    avg_dist_values.append(avg_dist_cosine_circle)
    continuity_values.append(continuity_circle)
    neighborhood_hit_values.append(neighborhood_hit_circle)
    shepard_goodness_values.append(shepard_goodness_circle)
    distance_consistency_values.append(distance_consistency_circle)

    for name, ld_data_other in other_projections:
        try:
            ld_data_other = convert_projection_to_array(ld_data_other)
        except TypeError as e:
            print(f"Skipping {name}: {e}")
            continue

        ld_cosine_dist_other = cosine_distances(ld_data_other)

        stress_cosine_other = calculate_stress(hd_cosine_dist, ld_cosine_dist_other)
        correlation_cosine_other = calculate_correlation(hd_cosine_dist, ld_cosine_dist_other)
        silhouette_cosine_other = silhouette_score(ld_data_other, labels, metric='cosine')
        trust_cosine_other = calculate_trustworthiness(hd_data, ld_data_other, distance='cosine')
        avg_dist_cosine_other = calculate_average_distance(ld_data_other, distance='cosine')
        continuity_other = calculate_continuity(hd_data, ld_data_other)
        neighborhood_hit_other = calculate_neighborhood_hit(hd_data, ld_data_other, labels)
        shepard_goodness_other = calculate_shepard_goodness(hd_cosine_dist, ld_cosine_dist_other)
        distance_consistency_other = calculate_distance_consistency(hd_data, ld_data_other)

        technique_names.append(name)
        stress_values.append(stress_cosine_other)
        correlation_values.append(correlation_cosine_other)
        silhouette_values.append(silhouette_cosine_other)
        trust_values.append(trust_cosine_other)
        avg_dist_values.append(avg_dist_cosine_other)
        continuity_values.append(continuity_other)
        neighborhood_hit_values.append(neighborhood_hit_other)
        shepard_goodness_values.append(shepard_goodness_other)
        distance_consistency_values.append(distance_consistency_other)

        print(f"Finished with {name}")

    print(f"{'Metric':<25} | " + " | ".join(f"{name:>20}" for name in technique_names))
    print("-" * (25 + len(technique_names) * 24))
    print(f"{'Cosine - Stress':<25} | " + " | ".join(f"{val:20.3f}" for val in stress_values))
    print(f"{'Cosine - Correlation':<25} | " + " | ".join(f"{val:20.3f}" for val in correlation_values))
    print(f"{'Cosine - Silhouette':<25} | " + " | ".join(f"{val:20.3f}" for val in silhouette_values))
    print(f"{'Cosine - Trustworthiness':<25} | " + " | ".join(f"{val:20.3f}" for val in trust_values))
    print(f"{'Cosine - Avg. Distance':<25} | " + " | ".join(f"{val:20.3f}" for val in avg_dist_values))
    print(f"{'Continuity':<25} | " + " | ".join(f"{val:20.3f}" for val in continuity_values))
    print(f"{'Neighborhood Hit':<25} | " + " | ".join(f"{val:20.3f}" for val in neighborhood_hit_values))
    print(f"{'Shepard Goodness':<25} | " + " | ".join(f"{val:20.3f}" for val in shepard_goodness_values))
    print(f"{'Distance Consistency':<25} | " + " | ".join(f"{val:20.3f}" for val in distance_consistency_values))

    metrics = {
        'technique_names': technique_names,
        'stress_values': stress_values,
        'correlation_values': correlation_values,
        'silhouette_values': silhouette_values,
        'trust_values': trust_values,
        'avg_dist_values': avg_dist_values,
        'continuity_values': continuity_values,
        'neighborhood_hit_values': neighborhood_hit_values,
        'shepard_goodness_values': shepard_goodness_values,
        'distance_consistency_values': distance_consistency_values
    }

    return metrics
