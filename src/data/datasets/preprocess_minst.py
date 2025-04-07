import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

# Define the directory for preprocessed data
PREPROCESSED_DATA_DIR = os.path.join("src", "data", "datasets", "preprocessed")

def scale_to_unit_range(data):
    """
    Scales the data to be within the range [-1, 1].
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def preprocess_mnist():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    images = np.concatenate([train_images, test_images], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    # Flatten images and scale
    flat_images = images.reshape(images.shape[0], -1)
    scaled_images = scale_to_unit_range(flat_images)

    # Create a DataFrame for easier integration into the pipeline
    mnist_df = pd.DataFrame(scaled_images, columns=[f'pixel_{i}' for i in range(scaled_images.shape[1])])
    mnist_df['target'] = labels

    # Save to CSV
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    file_path = os.path.join(PREPROCESSED_DATA_DIR, "mnist_scaled.csv")
    mnist_df.to_csv(file_path, index=False)
    print(f"[INFO] MNIST dataset preprocessed and saved to {file_path}")

# Run the preprocessing
preprocess_mnist()
