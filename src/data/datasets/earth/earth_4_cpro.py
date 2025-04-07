import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import dual_annealing
from mpl_toolkits.mplot3d import Axes3D

# File paths
data_file = "earth/earth_data.csv"
output_dir = "earth"
os.makedirs(output_dir, exist_ok=True)

# Constants
RADIUS = 6371.0  # Earth radius for scaling
SAMPLE_SIZE = 80  # Number of points to sample

def load_earth_data(data_file):
    """Loads the Earth dataset from a CSV file."""
    df = pd.read_csv(data_file)
    features = df[["x", "y", "z"]].values
    labels = df["continent"].values
    return features, labels, df

def normalize_features(features):
    """Normalize features to the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(features)

def stratified_sample(features, labels, sample_size):
    """Perform stratified sampling to ensure all continents are represented."""
    unique_labels = np.unique(labels)
    sampled_features, sampled_labels = [], []
    per_class_sample = max(1, sample_size // len(unique_labels))
    for label in unique_labels:
        class_indices = np.where(labels == label)[0]
        if len(class_indices) > per_class_sample:
            sampled_indices = np.random.choice(class_indices, per_class_sample, replace=False)
        else:
            sampled_indices = class_indices
        sampled_features.append(features[sampled_indices])
        sampled_labels.append(labels[sampled_indices])
    return np.vstack(sampled_features), np.hstack(sampled_labels)

def plot_3d_data(features, labels):
    """Plots the original 3D data on a sphere and saves it."""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        ax.scatter(
            features[indices, 0],
            features[indices, 1],
            features[indices, 2],
            label=label,
            s=6,
        )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Original 3D Spherical Data")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()
    save_path = os.path.join(output_dir, "original_3d_data.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def plot_projection(projection, labels, title, filename):
    """Plots the 2D projection with labels and saves it."""
    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        indices = np.where(labels == label)
        plt.scatter(projection[indices, 0], projection[indices, 1], label=label, s=25)
    plt.title(title)
    plt.xlim(-1.5, 1.5)  # Set x-axis scale
    plt.ylim(-1.5, 1.5)  # Set y-axis scale
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot: {save_path}")

def pca_projection(features, labels):
    """Projects the features using PCA and saves the plot."""
    pca = PCA(n_components=2)
    pca_proj = pca.fit_transform(features)
    plot_projection(pca_proj, labels, "PCA Projection", "pca_projection.png")

def mds_projection(features, labels):
    """Projects the features using MDS and saves the plot."""
    mds = MDS(n_components=2, random_state=42)
    mds_proj = mds.fit_transform(features)
    plot_projection(mds_proj, labels, "MDS Projection", "mds_projection.png")

def run_cpro(features, max_iterations=100):
    """Runs cPro circular projection using simulated annealing."""
    n = features.shape[0]

    # Compute high-dimensional cosine distances
    def compute_hd_distances(points):
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / norm
        cosine_distances = 1 - np.dot(normalized_points, normalized_points.T)
        return cosine_distances / 2

    hd_distances = compute_hd_distances(features)

    # Low-dimensional distance matrix calculation
    def compute_ld_distances(ld_points):
        ld_distances = np.abs(ld_points[:, None] - ld_points[None, :])
        return np.minimum(ld_distances, 1 - ld_distances)

    # Loss function
    def loss(ld_points):
        ld_distances = compute_ld_distances(ld_points)
        return np.sum(np.abs(hd_distances - (2 * ld_distances))) / 2

    # Bounds for optimization
    bounds = [(0, 1) for _ in range(n)]

    # Optimize using simulated annealing
    result = dual_annealing(loss, bounds, maxiter=max_iterations)
    ld_points = result.x

    # Convert to Cartesian coordinates for circular layout
    theta = ld_points * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    return np.column_stack((x, y))

def plot_cpro(features, labels):
    """Runs cPro, projects the features, and saves the circular plot."""
    cpro_proj = run_cpro(features, max_iterations=50)
    plot_projection(cpro_proj, labels, "cPro Circular Projection", "cpro_projection.png")

def main():
    """Main function to process and plot all projections."""
    features, labels, _ = load_earth_data(data_file)

    # Normalize features
    print("Normalizing features...")
    features = normalize_features(features)

    # Plot original 3D data
    print("Plotting original 3D data...")
    plot_3d_data(features, labels)

    # Stratified sampling
    print(f"Original data size: {len(features)}")
    features, labels = stratified_sample(features, labels, SAMPLE_SIZE)
    print(f"Sampled data size: {len(features)}")

    # PCA Projection
    print("Running PCA projection...")
    pca_projection(features, labels)

    # MDS Projection
    print("Running MDS projection...")
    mds_projection(features, labels)

    # cPro Projection
    print("Running cPro projection...")
    plot_cpro(features, labels)

    print(f"All projections are saved in {output_dir}.")

if __name__ == "__main__":
    main()
