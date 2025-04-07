import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import dual_annealing
from gensim.models import KeyedVectors

# Constants
WORDS = ["gay", "daft", "sweet", "flaunting", "cheerful", "tasteful",
         "pleasant", "witty", "bright", "bisexual", "homosexual", "lesbian"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTTEXT_FILE = os.path.join(CURRENT_DIR, "wiki-news-300d-1M.vec")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "visualization_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 1: Load FastText Embeddings
def load_fasttext_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FastText embeddings not found at {file_path}. Please ensure the file exists.")
    model = KeyedVectors.load_word2vec_format(file_path, binary=False)
    return model

# Step 2: Get Word Vectors
def get_word_vectors(words, model):
    vectors = []
    valid_words = []
    for word in words:
        if word in model:
            vectors.append(model[word])
            valid_words.append(word)
        else:
            print(f"Word '{word}' not found in embeddings. Skipping.")
    return np.array(vectors), valid_words

# Step 3: Compute Global Origin
def compute_global_origin(vectors):
    return np.mean(vectors, axis=0)

# Step 4: cPro Implementation
def run_cpro(vectors, max_iterations=100):
    n = vectors.shape[0]

    # Compute high-dimensional cosine distances
    def compute_hd_distances(points):
        norm = np.linalg.norm(points, axis=1, keepdims=True)
        normalized_points = points / norm
        cosine_distances = 1 - np.dot(normalized_points, normalized_points.T)
        return cosine_distances / 2

    hd_distances = compute_hd_distances(vectors)

    # Low-dimensional distance matrix calculation
    def compute_ld_distances(ld_points):
        ld_distances = np.linalg.norm(ld_points[:, None] - ld_points[None, :], axis=-1)
        return ld_distances

    # Loss function
    def loss(ld_points):
        ld_points = ld_points.reshape(-1, 2)
        ld_distances = compute_ld_distances(ld_points)
        return np.sum(np.abs(hd_distances - ld_distances))

    # Initial layout (uniform random)
    initial_layout = np.random.uniform(-1, 1, (n, 2))

    # Bounds for optimization
    bounds = [(-1, 1)] * (n * 2)

    # Optimize using dual annealing
    result = dual_annealing(loss, bounds, maxiter=max_iterations)
    optimized_layout = result.x.reshape(-1, 2)
    return optimized_layout

# Step 5: Visualize Results
def visualize_cpro_with_similarity(labels, projection, similarities, output_file):
    # Normalize to enforce radial projection
    radii = np.linalg.norm(projection, axis=1)
    projection = projection / radii[:, None]  # Normalize to unit circle
    
    # Shift points based on similarity to global origin
    projection = projection * (1 - similarities[:, None])

    # Create the plot
    plt.figure(figsize=(10, 10))  # Square plot with equal width and height
    plt.scatter(projection[:, 0], projection[:, 1], s=150, c='blue', alpha=0.7, edgecolors='k')
    
    # Add a central node for the global origin
    plt.scatter(0, 0, s=200, facecolors='none', edgecolors='black', label="Global Origin")
    plt.text(0, -0.05, "Origin", fontsize=10, ha='center', va='center', color='black')

    # Add labels below points
    for label, (x, y) in zip(labels, projection):
        plt.text(
            x, y - 0.05, label, fontsize=10, ha='center', va='center',
            color='black'
        )

    # Adjust axis limits for better visibility
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    # Remove grid
    plt.grid(False)

    # Add title and labels
    plt.title(f"Radial cPro Visualization for Semantic Space of Selected Words", fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    # Ensure the aspect ratio is equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Save and display the plot
    plt.legend()
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Loading FastText embeddings...")
    fasttext_model = load_fasttext_embeddings(FASTTEXT_FILE)

    print("Extracting word vectors...")
    word_vectors, valid_words = get_word_vectors(WORDS, fasttext_model)

    print("Computing global origin...")
    global_origin = compute_global_origin(word_vectors)

    print("Calculating similarities to global origin...")
    similarities = cosine_similarity(word_vectors, [global_origin]).flatten()

    print("Running cPro...")
    projection = run_cpro(word_vectors)

    print("Visualizing results...")
    output_file = os.path.join(OUTPUT_DIR, "cpro_semantic_space.png")
    visualize_cpro_with_similarity(valid_words, projection, similarities, output_file)
    print(f"Visualization saved to {output_file}.")
