import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

# Define directory paths
DATA_DIR = os.path.join('src', 'data', 'datasets', 'preprocessed')
RAW_FILE = 'yelp.csv'
FILENAME = 'yelp_reviews_vectorized.csv'
MAX_FEATURES = 300

def ensure_data_dir():
    """Ensures the preprocessed data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def scale_to_unit_range(data):
    """Scales data to be within the range [-1, 1]."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(data)

def load_yelp_data():
    """Loads the Yelp Reviews dataset and processes the review text and labels."""
    # Load the data from the same directory as this script
    file_path = os.path.join(os.path.dirname(__file__), RAW_FILE)
    
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at {file_path}.")
        return pd.DataFrame()
    
    # Load the data
    yelp_data = pd.read_csv(file_path)
    
    # We assume the dataset has columns 'text' for reviews and 'stars' for labels
    yelp_data = yelp_data[['text', 'stars']].dropna()
    
    # Rename columns for consistency
    yelp_data.rename(columns={'text': 'text', 'stars': 'target'}, inplace=True)
    print(f"Total Yelp reviews loaded: {len(yelp_data)}")
    return yelp_data

def vectorize_and_save(dataframe, filename, max_features=MAX_FEATURES):
    """Vectorizes text data, scales, and saves it to CSV."""
    if dataframe.empty:
        print("Warning: No data found in the Yelp Reviews dataset.")
        return
    
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
        print(f"Yelp Reviews dataset saved to {file_path}")
    else:
        print("Warning: No valid vocabulary found after vectorization.")

def preprocess_yelp_reviews():
    """Loads, processes, and saves the Yelp Reviews dataset."""
    yelp_data = load_yelp_data()
    vectorize_and_save(yelp_data, FILENAME)

# Run the preprocessing if this script is executed directly
if __name__ == "__main__":
    preprocess_yelp_reviews()
