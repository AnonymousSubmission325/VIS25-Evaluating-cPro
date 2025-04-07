import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Define the directory for preprocessed data
PREPROCESSED_DATA_DIR = os.path.join("src", "data", "datasets", "preprocessed")

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def load_preprocessed_data(filename, sample_size=None):
    """
    Loads a preprocessed dataset from a CSV file, optionally samples it, 
    and scales its features to the range [-1, 1].
    
    Parameters:
    -----------
    filename : str
        Name of the CSV file in the preprocessed data directory.
    sample_size : int, optional
        Number of rows to sample. Loads full dataset if None.

    Returns:
    --------
    formatted_data : pd.DataFrame
        The preprocessed dataset with features scaled and target column intact.
    """
    file_path = os.path.join(PREPROCESSED_DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"[ERROR] File {filename} not found in preprocessed data directory.")
        return pd.DataFrame()

    # Load the data
    data = pd.read_csv(file_path)
    
    if 'target' not in data.columns:
        print(f"[ERROR] Target column not found in {filename}.")
        return pd.DataFrame()

    # Optional sampling
    if sample_size and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=0)
        print(f"[INFO] Sampled {sample_size} rows from {filename}")

    # Scale features and format data
    features = data.drop(columns=['target']).values
    features = scale_to_unit_range(features)
    labels = data['target'].values
    
    formatted_data = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
    formatted_data['target'] = labels
    
    return formatted_data

def import_reuters():
    return load_preprocessed_data("reuters_vectorized.csv")

def import_imdb():
    return load_preprocessed_data("imdb_reviews_vectorized.csv")

def import_20newsgroups():
    return load_preprocessed_data("20newsgroups_vectorized.csv")

def import_ag_news():
    return load_preprocessed_data("ag_news_vectorized.csv", sample_size=10000)

def import_yelp():
    return load_preprocessed_data("yelp_reviews_vectorized.csv", sample_size=25)

def import_iris():
    return load_preprocessed_data("iris_scaled.csv")

def import_penguins():
    return load_preprocessed_data("penguins.csv")

def import_breast_cancer():
    return load_preprocessed_data("breast_cancer_scaled.csv")

def import_dbpedia():
    return load_preprocessed_data("dbpedia_vectorized.csv")

def import_wine():
    return load_preprocessed_data("wine_scaled.csv", sample_size=80)

def load_datasets():
    """
    Loads all datasets from the preprocessed folder and returns them in a dictionary.
    """
    datasets = {
        # 'reuters': import_reuters(),
        # 'imdb': import_imdb(),
        # '20newsgroups': import_20newsgroups(),
        # 'ag_news': import_ag_news(),
        # 'yelp': import_yelp(),
        # 'iris': import_iris(),
        # 'penguins': import_penguins(),
        # 'breast_cancer': import_breast_cancer(),
        # 'dbpedia': import_dbpedia(),
        'wine': import_wine()
    }
    return datasets


# Directory containing artificial datasets
ARTIFICIAL_DATA_DIR = os.path.join("src", "data", "datasets", "artificial")

def load_artificial_dataset(filename):
    """
    Loads a single artificial dataset from a CSV file and ensures the correct format.
    """
    file_path = os.path.join(ARTIFICIAL_DATA_DIR, filename)
    if not os.path.exists(file_path):
        print(f"[ERROR] File {filename} not found in artificial data directory.")
        return pd.DataFrame()

    data = pd.read_csv(file_path)
    if 'target' not in data.columns:
        print(f"[ERROR] Target column not found in {filename}.")
        return pd.DataFrame()

    return data

# Directory path to artificial datasets
ARTIFICIAL_DATASET_DIR = "src/data/datasets/artificial"

def load_artificial_datasets():
    """
    Loads all artificial datasets from the specified directory and returns them in a dictionary.

    Returns:
    --------
    dict
        Dictionary where keys are dataset names (from filenames) and values are DataFrames with 'x', 'y', and 'target' columns.
    """
    datasets = {}
    
    # List of datasets to load
    dataset_files = {
        # '2dsphere': "2dsphere_cPro.csv",
        # '3dsphere': "3dsphere_cPro.csv",
        # '4dsphere': "4dsphere_cPro.csv",
        # 'blobs_3d_v1': "blobs_3d_v1_cPro.csv",
        # 'blobs_3d_v2': "blobs_3d_v2_cPro.csv",
        # 'blobs_3d_v3': "blobs_3d_v3_cPro.csv",
        # 'blobs_3d_v4': "blobs_3d_v4_cPro.csv",
        # 'citations': "citations_cPro.csv",
        # 'concentric_circles': "concentric_circles_cPro.csv",
        # 'iris': "iris_cPro.csv",
        # 'penguins': "penguins_cPro.csv",
        # 's_curve': "s_curve_cPro.csv",
        # 'three_circles': "three_circles_cPro.csv",
        # 'torus': "torus_cPro.csv",
        # 'unbalanced': "unbalanced_cPro.csv",
    }
    
    for name, filename in dataset_files.items():
        file_path = os.path.join(ARTIFICIAL_DATASET_DIR, filename)
        
        if not os.path.exists(file_path):
            print(f"[WARNING] File {filename} not found in {ARTIFICIAL_DATASET_DIR}. Skipping.")
            continue
        
        # Load the dataset and keep only the 'x', 'y', and 'target' columns
        df = pd.read_csv(file_path)
        if {'x', 'y', 'target'}.issubset(df.columns):
            datasets[name] = df[['x', 'y', 'target']]
        else:
            print(f"[ERROR] Expected columns ('x', 'y', 'target') not found in {filename}. Skipping this file.")

    return datasets

# Example usage
if __name__ == "__main__":
    artificial_datasets = load_artificial_datasets()
    for name, df in artificial_datasets.items():
        print(f"{name} dataset loaded with shape: {df.shape}")


# Example usage
if __name__ == "__main__":
    datasets = load_datasets()
    for name, data in datasets.items():
        if not data.empty:
            print(f"{name} dataset loaded successfully.")
            print(data.head())
        else:
            print(f"{name} dataset could not be loaded.")
