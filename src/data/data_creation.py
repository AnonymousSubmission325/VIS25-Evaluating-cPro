import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.preprocessing import MinMaxScaler

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def create_sample_data():
    # Set parameters
    n_points = 200  # Total number of data points
    n_clusters = 4  # Number of clusters

    # Generate 3D data with 4 blobs
    data, labels = make_blobs(n_samples=n_points, centers=n_clusters, n_features=3, random_state=777)

    # Scale the data to be within [-1, 1]
    data = scale_to_unit_range(data)

    # Convert the scaled data to a pandas DataFrame
    sample_data = pd.DataFrame(data, columns=['x', 'y', 'z'])

    # Add the target variable to the DataFrame
    sample_data['target'] = labels

    return sample_data, labels

def create_sample_data_2d():
    # Set parameters for 2D dataset
    n_points = 200  # Total number of data points
    n_clusters = 3  # Number of clusters

    # Generate 2D data with 3 blobs
    data, labels = make_blobs(n_samples=n_points, centers=n_clusters, n_features=2, random_state=777)

    # Scale the data to be within [-1, 1]
    data = scale_to_unit_range(data)

    # Convert the scaled data to a pandas DataFrame
    sample_data = pd.DataFrame(data, columns=['x', 'y'])

    # Add the target variable to the DataFrame
    sample_data['target'] = labels

    return sample_data, labels

def create_classification_data(n_points=200, n_features=5, n_classes=3):
    """
    Generates a classification dataset with the specified number of samples, features, and classes.
    """
    data, labels = make_classification(
        n_samples=n_points,
        n_features=n_features,
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=2,
        n_classes=n_classes,
        random_state=777
    )
    
    # Scale the data to be within [-1, 1]
    data = scale_to_unit_range(data)

    # Convert to DataFrame and include the 'target' column
    sample_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    sample_data['target'] = labels

    return sample_data

def create_blobs_data(n_samples=200, n_features=3, n_clusters=4):
    """
    Generates a synthetic 3D blobs dataset for clustering.
    """
    data, labels = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=777)

    # Scale the data to be within [-1, 1]
    data = scale_to_unit_range(data)

    # Convert the scaled data to a pandas DataFrame
    sample_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    sample_data['target'] = labels

    return sample_data

def create_high_dimensional_data(n_points=500, n_features=10, n_classes=4):
    """
    Generates a high-dimensional classification dataset with the specified number of samples, features, and classes.
    """
    data, labels = make_classification(
        n_samples=n_points,
        n_features=n_features,
        n_informative=6,  # Number of informative features
        n_redundant=2,    # Number of redundant features
        n_clusters_per_class=2,
        n_classes=n_classes,
        random_state=777
    )
    
    # Scale the data to be within [-1, 1]
    data = scale_to_unit_range(data)

    # Convert to DataFrame and include the 'target' column
    sample_data = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    sample_data['target'] = labels

    return sample_data

def create_multiple_datasets():
    """
    Creates multiple datasets for evaluation.
    """
    return {
        # 'blobs_3d': create_blobs_data(n_samples=200, n_features=3, n_clusters=4),
        # 'high_dimensional': create_high_dimensional_data(n_points=500, n_features=10, n_classes=4),
        # Add more datasets as needed
    }