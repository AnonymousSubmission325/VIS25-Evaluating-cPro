import os
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = os.path.join('src', 'data', 'datasets', 'preprocessed')
MAX_FEATURES = 300  # Max features for TF-IDF

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def fetch_20newsgroups_data():
    """Fetches and preprocesses the 20 Newsgroups dataset."""
    print("Fetching 20 Newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data
    labels = newsgroups.target
    label_names = newsgroups.target_names
    print(f"Fetched {len(documents)} documents.")
    
    return pd.DataFrame({'text': documents, 'target': labels}), label_names

def vectorize_and_save(dataframe, filename, label_names, max_features=MAX_FEATURES):
    """Vectorizes text data, scales, and saves it to CSV."""
    if dataframe.empty:
        print("Warning: No documents remain after filtering. Check document contents and filtering criteria.")
        return  # Exit function if no data remains

    dataframe['text'] = dataframe['text'].astype(str)
    dataframe = dataframe[dataframe['text'].str.strip().astype(bool)]
    print(f"Documents after filtering empty texts: {len(dataframe)}")

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(dataframe['text'])

    if tfidf_matrix.shape[1] > 0:  # Ensure there are features after vectorization
        features = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'feature_{i}' for i in range(tfidf_matrix.shape[1])])
        features['target'] = dataframe['target'].values
        feature_columns = [col for col in features.columns if col != 'target']
        features[feature_columns] = scale_to_unit_range(features[feature_columns])

        ensure_data_dir()
        file_path = os.path.join(DATA_DIR, filename)
        features.to_csv(file_path, index=False)
        print(f"20 Newsgroups data saved to {file_path}")
    else:
        print("Warning: No valid vocabulary found after vectorization.")

def load_20newsgroups_dataset():
    """Loads the 20 Newsgroups dataset, using preprocessed data if available."""
    filename = '20newsgroups_vectorized.csv'
    filepath = os.path.join(DATA_DIR, filename)

    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    
    newsgroups_data, label_names = fetch_20newsgroups_data()
    vectorize_and_save(newsgroups_data, filename, label_names)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        print("Error: Failed to create the processed dataset.")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    newsgroups_dataset = load_20newsgroups_dataset()
    if not newsgroups_dataset.empty:
        print(newsgroups_dataset.head())
    else:
        print("No data loaded.")
