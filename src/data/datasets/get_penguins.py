import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.datasets import fetch_openml

DATA_DIR = os.path.join('src', 'data', 'datasets', 'preprocessed')

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def encode_labels(dataframe, column_name):
    """Encodes categorical labels into numerical values."""
    label_encoder = LabelEncoder()
    dataframe[column_name] = label_encoder.fit_transform(dataframe[column_name])
    return dataframe, label_encoder

def fetch_penguins_data():
    """Fetches and preprocesses the Penguins dataset."""
    print("Fetching Penguins dataset...")
    try:
        data = fetch_openml(name='penguins', version=1, as_frame=True)
    except Exception as e:
        print(f"Error fetching the dataset: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure
    
    df = data.frame
    df = df.dropna()  # Drop rows with missing values
    print(f"Fetched {len(df)} rows after removing missing values.")
    print("Columns found in the dataset:", df.columns.tolist())  # Debug: Print column names
    return df

def preprocess_and_save_penguins(df, filename):
    """Preprocesses the Penguins dataset and saves it to CSV."""
    # Updated to match the dataset's actual column names
    numerical_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
    categorical_columns = ['species', 'island', 'sex']

    # Sanity check: Ensure numerical columns exist
    for col in numerical_columns:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in dataset.")

    # Encode categorical columns
    label_encoders = {}
    for col in categorical_columns:
        if col in df.columns:
            df, label_encoder = encode_labels(df, col)
            label_encoders[col] = label_encoder
        else:
            print(f"Warning: Column '{col}' not found. Skipping label encoding.")

    # Scale numerical features
    df[numerical_columns] = scale_to_unit_range(df[numerical_columns])

    # Add target column (species) for classification
    if 'species' not in df.columns:
        raise KeyError("Column 'species' not found in dataset for target assignment.")
    df['target'] = df['species']  # Use encoded 'species' as the target

    # Save to CSV
    ensure_data_dir()
    file_path = os.path.join(DATA_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Penguins data saved to {file_path}")

def load_penguins_dataset():
    """Loads the Penguins dataset, using preprocessed data if available."""
    filename = 'penguins_preprocessed.csv'
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    df = fetch_penguins_data()
    if not df.empty:
        preprocess_and_save_penguins(df, filename)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print("Error: Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    penguins_dataset = load_penguins_dataset()
    if not penguins_dataset.empty:
        print(penguins_dataset.head())
    else:
        print("No data loaded.")
