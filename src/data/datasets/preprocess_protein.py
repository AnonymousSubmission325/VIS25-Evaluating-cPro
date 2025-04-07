import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define constants
DATA_DIR = os.path.join('src', 'data', 'datasets', 'preprocessed')
PROTEIN_FILENAME = 'protein_embeddings.csv'

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def preprocess_protein_data(input_path, output_filename=PROTEIN_FILENAME):
    """
    Loads and preprocesses protein embedding data.

    Parameters:
    ----------
    input_path : str
        Path to the raw protein data file (e.g., embeddings from AlphaFold or Rosetta).
    output_filename : str
        Name of the output CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    print(f"Loading protein data from {input_path}...")
    protein_data = pd.read_csv(input_path)

    # Ensure necessary columns exist
    required_columns = [col for col in protein_data.columns if col.startswith('feature_')]
    if 'label' not in protein_data.columns or not required_columns:
        print("Error: The input file must contain 'label' and feature columns (e.g., 'feature_0').")
        return

    # Scale feature columns
    print(f"Scaling {len(required_columns)} feature columns to the range [-1, 1]...")
    protein_data[required_columns] = scale_to_unit_range(protein_data[required_columns])

    # Save preprocessed data
    ensure_data_dir()
    file_path = os.path.join(DATA_DIR, output_filename)
    protein_data.to_csv(file_path, index=False)
    print(f"Preprocessed protein data saved to {file_path}")

def load_protein_dataset(input_path):
    """Loads the protein dataset, using preprocessed data if available."""
    filepath = os.path.join(DATA_DIR, PROTEIN_FILENAME)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    preprocess_protein_data(input_path)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print("Error: Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    raw_protein_data_path = 'protein_embeddings.csv'  # Replace with your actual file path
    protein_dataset = load_protein_dataset(raw_protein_data_path)
    if not protein_dataset.empty:
        print(protein_dataset.head())
    else:
        print("No data loaded.")
