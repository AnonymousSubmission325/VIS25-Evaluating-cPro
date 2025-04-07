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
    similarities = np.array([distances[word] for word in labels])
    return labels, vectors, similarities

# Step 4: Spherical PCA Projection with Radial Similarity
def run_spca_with_similarity(vectors, similarities, min_radius=0.2):
    # PCA projection to 2D
    pca = PCA(n_components=2)
    projected = pca.fit_transform(vectors)

    # Normalize directions
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms[norms == 0] = 1
    directions = projected / norms

    # Normalize similarity to [0, 1]
    sim_min = np.min(similarities)
    sim_max = np.max(similarities)
    if sim_max - sim_min > 1e-6:
        normalized_sim = (similarities - sim_min) / (sim_max - sim_min)
    else:
        normalized_sim = np.zeros_like(similarities)

    # Invert and scale to [min_radius, 1.0]
    radii = min_radius + (1 - min_radius) * (1 - normalized_sim)

    # Apply to directions
    spherical_scaled_projection = directions * radii[:, None]
    return spherical_scaled_projection

# Visualization
def visualize_spca_scaled(labels, projection, output_file):
    plt.figure(figsize=(10, 10))
    plt.scatter(projection[:, 0], projection[:, 1], s=150, facecolors='black', edgecolors='black', alpha=0.7)
    for label, (x, y) in zip(labels, projection):
        plt.text(x, y - 0.05, label, fontsize=10, ha='center', va='top')
    plt.scatter([0], [0], s=300, facecolors='none', edgecolors='black', linewidths=1.5)
    plt.text(0, -0.08, TARGET_WORD, fontsize=12, ha='center', va='top', fontweight='bold')
    plt.title(f"sPCA with Similarity-Encoded Radius for '{TARGET_WORD}'", fontsize=14)
    plt.xlabel("Component 1 (directional)", fontsize=12)
    plt.ylabel("Component 2 (directional)", fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
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
    labels, vectors, similarities = prepare_data(distances, fasttext_model)

    print("Running sPCA with similarity-based radius...")
    projection = run_spca_with_similarity(vectors, similarities)

    print("Visualizing sPCA results...")
    output_file = os.path.join(OUTPUT_DIR, f"{TARGET_WORD}_spca_similarity_radius_visualization.png")
    visualize_spca_scaled(labels, projection, output_file)
    print(f"Visualization saved to {output_file}.")
