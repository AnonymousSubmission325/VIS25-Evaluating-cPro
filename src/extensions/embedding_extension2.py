import os
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTTEXT_FILE = os.path.join(CURRENT_DIR, "wiki-news-300d-1M.vec")
TARGET_WORD = "spring"
RELATED_TERMS = [
    "spring", "winter", "summer", "autumn", "season", "bloom", "flower",  # Seasonal
    "river", "fountain", "geyser", "water", "creek", "well"    # Water-related
]

OUTPUT_DIR = os.path.join(CURRENT_DIR, "visualization_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load FastText Embeddings
def load_fasttext_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FastText embeddings not found at {file_path}. Please ensure the file exists.")
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)
    return model

# Step 2: Get Similarity Distances
def get_distances_from_target(word, related_terms, model):
    if word not in model:
        raise ValueError(f"Word '{word}' not found in embeddings.")
    distances = {}
    for term in related_terms:
        if term in model:
            distance = cosine_similarity([model[word]], [model[term]])[0][0]
            distances[term] = distance
        else:
            print(f"Word '{term}' not found in embeddings. Skipping.")
    return distances

# Step 3: Prepare Data
def prepare_data(distances, model):
    labels = list(distances.keys())
    vectors = np.array([model[word] for word in labels])
    return labels, vectors

# Step 4: PCA Projection
def run_pca(vectors):
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors)

# Visualization
def visualize_pca(labels, projection, output_file):
    plt.figure(figsize=(10, 10))
    plt.scatter(projection[:, 0], projection[:, 1], s=150, facecolors='black', edgecolors='black', alpha=0.7)
    for label, (x, y) in zip(labels, projection):
        plt.text(x, y - 0.05, label, fontsize=10, ha='center', va='top')
    plt.title(f"PCA Visualization for '{TARGET_WORD}' Related Terms", fontsize=14)
    plt.xlabel("Principal Component 1", fontsize=12)
    plt.ylabel("Principal Component 2", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)  # Removed the grid
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Loading FastText embeddings...")
    fasttext_model = load_fasttext_embeddings(FASTTEXT_FILE)

    print(f"Calculating distances for related terms of '{TARGET_WORD}'...")
    distances = get_distances_from_target(TARGET_WORD, RELATED_TERMS, fasttext_model)
    for term, distance in distances.items():
        print(f"Similarity between '{TARGET_WORD}' and '{term}': {distance:.4f}")
    labels, vectors = prepare_data(distances, fasttext_model)

    print("Running PCA...")
    projection = run_pca(vectors)

    print("Visualizing PCA results...")
    output_file = os.path.join(OUTPUT_DIR, f"{TARGET_WORD}_pca_related_terms_visualization.png")
    visualize_pca(labels, projection, output_file)
    print(f"Visualization saved to {output_file}.")
