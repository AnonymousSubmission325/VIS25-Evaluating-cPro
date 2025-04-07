import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Define the directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'preprocessed')
RAW_DATA_DIR = os.path.join(BASE_DIR, 'reuters', 'reuters', 'training')
MAX_FEATURES = 300  # Number of features for TF-IDF

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    print(f"[INFO] Preprocessed data directory checked/created at {DATA_DIR}")

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def gather_reuters_data():
    """Reads each file in the RAW_DATA_DIR as a separate document and labels it."""
    documents, labels = [], []

    print(f"[INFO] Reading files from RAW_DATA_DIR: {RAW_DATA_DIR}")
    
    if os.path.exists(RAW_DATA_DIR):
        for filename in os.listdir(RAW_DATA_DIR):
            file_path = os.path.join(RAW_DATA_DIR, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        document_text = f.read().strip()
                        documents.append(document_text)
                        labels.append('training')  # Generic label; adjust if specific labels are required
                    print(f"[INFO] Loaded document: {filename}")
                except Exception as e:
                    print(f"[ERROR] Failed to read {file_path}: {e}")
            else:
                print(f"[WARNING] Skipping non-file item in directory: {file_path}")
    else:
        print(f"[ERROR] RAW_DATA_DIR does not exist: {RAW_DATA_DIR}")

    print(f"[INFO] Total documents collected: {len(documents)}")
    return pd.DataFrame({'text': documents, 'target': labels})

def vectorize_and_save(dataframe, filename, max_features=MAX_FEATURES):
    """Vectorizes text data using TF-IDF, scales it, and saves the resulting DataFrame to a CSV file."""
    if dataframe.empty:
        print("[WARNING] No documents to process after gathering. Exiting function.")
        return

    # Clean and filter out empty documents
    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe[dataframe['text'].str.strip().astype(bool)]
    print(f"[INFO] Documents after filtering empty texts: {len(dataframe)}")
    
    # Vectorize the documents using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dataframe['text'])
    
    if tfidf_matrix.shape[1] > 0:
        # Create DataFrame for TF-IDF features
        features = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'feature_{i}' for i in range(tfidf_matrix.shape[1])])
        features['target'] = dataframe['target'].values

        # Scale the features to [-1, 1] range
        feature_columns = [col for col in features.columns if col != 'target']
        features[feature_columns] = scale_to_unit_range(features[feature_columns])
        
        # Ensure the preprocessed data directory exists
        ensure_data_dir()
        
        # Save the processed data to a CSV file
        file_path = os.path.join(DATA_DIR, filename)
        features.to_csv(file_path, index=False)
        print(f"[INFO] Data saved to {file_path}")
    else:
        print("[WARNING] No valid vocabulary found after vectorization.")

def load_reuters_dataset():
    """Loads the Reuters dataset, using preprocessed data if available, or preprocesses if not."""
    filename = 'reuters_vectorized.csv'
    filepath = os.path.join(DATA_DIR, filename)

    # Check if preprocessed data already exists
    if os.path.exists(filepath):
        print(f"[INFO] Loading preprocessed data from {filepath}")
        return pd.read_csv(filepath)
    
    # If not, gather and preprocess the raw data
    print("[INFO] Preprocessed file not found. Gathering and processing raw data.")
    reuters_data = gather_reuters_data()
    vectorize_and_save(reuters_data, filename)
    
    # Load the newly created file if it exists
    if os.path.exists(filepath):
        print(f"[INFO] Successfully created and loaded the processed dataset from {filepath}")
        return pd.read_csv(filepath)
    else:
        print("[ERROR] Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    reuters_dataset = load_reuters_dataset()
    if not reuters_dataset.empty:
        print("[INFO] Sample data from the processed dataset:")
        print(reuters_dataset.head())
    else:
        print("[ERROR] No data loaded.")
