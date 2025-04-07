import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset  # Requires the `datasets` library

# Paths for data storage
DATA_DIR = os.path.join('datasets', 'preprocessed')
MAX_FEATURES = 300  # Adjust as necessary

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def gather_ag_news_data():
    """Downloads and formats AG News data."""
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news")
    documents = dataset['train']['text']
    labels = dataset['train']['label']

    print(f"Total documents collected: {len(documents)}")
    return pd.DataFrame({'text': documents, 'target': labels})

def vectorize_and_save(dataframe, filename, max_features=MAX_FEATURES):
    """Vectorizes text data, scales, and saves it to CSV."""
    if dataframe.empty:
        print("Warning: No documents remain after filtering.")
        return

    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe[dataframe['text'].str.strip().astype(bool)]
    print(f"Documents after filtering empty texts: {len(dataframe)}")

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

def load_ag_news_dataset():
    """Loads the AG News dataset, using preprocessed data if available."""
    filename = 'ag_news_vectorized.csv'
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    ag_news_data = gather_ag_news_data()
    vectorize_and_save(ag_news_data, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print("Error: Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    ag_news_dataset = load_ag_news_dataset()
    if not ag_news_dataset.empty:
        print(ag_news_dataset.head())
    else:
        print("No data loaded.")
