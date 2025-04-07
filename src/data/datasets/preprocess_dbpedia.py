import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset  # Requires the `datasets` library

# Directory for preprocessed data
DATA_DIR = os.path.join('datasets', 'preprocessed')
MAX_FEATURES = 300

def ensure_data_dir():
    """Ensure the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scale data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def gather_dbpedia_data():
    """Download DBpedia dataset and prepare it as a DataFrame."""
    dataset = load_dataset("dbpedia_14")
    texts = []
    labels = []

    for item in dataset['train']:
        texts.append(item['content'])
        labels.append(item['label'])

    return pd.DataFrame({'text': texts, 'target': labels})

def vectorize_and_save(dataframe, filename, max_features=MAX_FEATURES):
    """Vectorizes text data, scales it, and saves to CSV."""
    if dataframe.empty:
        print("Warning: No documents found after filtering.")
        return

    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe[dataframe['text'].str.strip().astype(bool)]
    
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dataframe['text'])

    if tfidf_matrix.shape[1] > 0:
        features = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'feature_{i}' for i in range(tfidf_matrix.shape[1])])
        features['target'] = dataframe['target'].values
        feature_columns = [col for col in features.columns if col != 'target']
        features[feature_columns] = scale_to_unit_range(features[feature_columns])

        ensure_data_dir()
        file_path = os.path.join(DATA_DIR, filename)
        features.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("Warning: No valid vocabulary found after vectorization.")

def load_dbpedia_dataset():
    """Loads the DBpedia dataset, using preprocessed data if available."""
    filename = 'dbpedia_vectorized.csv'
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    dbpedia_data = gather_dbpedia_data()
    vectorize_and_save(dbpedia_data, filename)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print("Error: Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    dbpedia_dataset = load_dbpedia_dataset()
    if not dbpedia_dataset.empty:
        print(dbpedia_dataset.head())
    else:
        print("No data loaded.")
