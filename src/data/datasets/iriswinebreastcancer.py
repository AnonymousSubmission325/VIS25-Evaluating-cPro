import os
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

# Define the preprocessed data directory
DATA_DIR = os.path.join('src', 'data', 'datasets', 'preprocessed')

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def save_dataset(data, target, filename):
    """Converts data to DataFrame, scales it, and saves it as CSV."""
    dataframe = pd.DataFrame(data)
    dataframe['target'] = target
    feature_columns = [col for col in dataframe.columns if col != 'target']
    dataframe[feature_columns] = scale_to_unit_range(dataframe[feature_columns])
    
    ensure_data_dir()
    file_path = os.path.join(DATA_DIR, filename)
    dataframe.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path}")

def preprocess_iris():
    """Loads and saves the Iris dataset."""
    iris = load_iris()
    save_dataset(iris.data, iris.target, 'iris_scaled.csv')

def preprocess_wine():
    """Loads and saves the Wine dataset."""
    wine = load_wine()
    save_dataset(wine.data, wine.target, 'wine_scaled.csv')

def preprocess_breast_cancer():
    """Loads and saves the Breast Cancer dataset."""
    cancer = load_breast_cancer()
    save_dataset(cancer.data, cancer.target, 'breast_cancer_scaled.csv')

# Run preprocessing for each dataset
if __name__ == "__main__":
    preprocess_iris()
    preprocess_wine()
    preprocess_breast_cancer()
