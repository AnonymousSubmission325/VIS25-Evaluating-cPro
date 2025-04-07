import os
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import dual_annealing

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FASTTEXT_FILE = os.path.join(CURRENT_DIR, "wiki-news-300d-1M.vec")
TARGET_WORD = "rock"
RELATED_TERMS = [
    "band", "guitar", "album", "concert", "rhythm", "lyrics", "melody",  # Music Genre
    "stone", "mountain", "cliff", "erosion", "boulder", "layer"  # Geological Formation
]

# TARGET_WORD = "spring"
# RELATED_TERMS = [
#     "winter", "summer", "autumn", "season", "bloom", "flower", "April"  # Seasonal
#     "stream", "river", "fountain", "geyser", "water", "creek", "well"
# ]
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

# Step 3: Prepare Data for cPro
def prepare_data(distances, model):
    labels = list(distances.keys())
    vectors = np.array([model[word] for word in labels])
    return labels, vectors

# Step 4: cPro Implementation
def run_cpro(vectors, max_iterations=4):
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

def visualize_cpro_with_similarity(labels, projection, similarities, output_file):
    # Normalize to enforce radial projection
    radii = np.linalg.norm(projection, axis=1)
    projection = projection / radii[:, None]  # Normalize to unit circle

    # Ensure proper radial symmetry by centering the projection
    mean_vector = np.mean(projection, axis=0)
    projection -= mean_vector  # Center around origin
    projection /= np.linalg.norm(projection, axis=1)[:, None]  # Re-normalize

    # Optional: Apply padding (scale slightly inside unit circle)
    padding_factor = 0.9
    projection *= padding_factor

    # Shift points towards the center based on similarity
    similarities = np.array(similarities)  # Ensure similarities are numpy array
    projection = projection * (1 - similarities[:, None])  # Adjust radius based on similarity

    # Create the plot
    plt.figure(figsize=(10, 10))  # Increased size for better visibility

    # Add a dashed circle representing the cPro radius
    circle = plt.Circle((0, 0), radius=padding_factor, color='gray', fill=False, linestyle='dashed')
    plt.gca().add_artist(circle)

    # Plot the central node (spring) with a smaller black-bordered circle
    plt.scatter([0], [0], s=300, facecolors='none', edgecolors='black', linewidths=1.5)
    plt.text(0, -0.08, TARGET_WORD, fontsize=12, ha='center', va='top', fontweight='bold')  # Text below the center

    # Plot other points with labels below
    plt.scatter(projection[:, 0], projection[:, 1], s=150, facecolors='black', edgecolors='black', alpha=0.7, label="Points")
    for label, (x, y) in zip(labels, projection):
        plt.text(
            x, y - 0.05, label, fontsize=10, ha='center', va='top'
        )

    # Adjust axis limits for better visibility
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)

    # Remove grid
    plt.grid(False)

    # Add title and labels
    plt.title(f"Radial cPro Visualization for '{TARGET_WORD}' Related Terms with Similarity Shift", fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)

    # Ensure the aspect ratio is equal
    plt.gca().set_aspect('equal', adjustable='box')

    # Save and display the plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()



# Main Execution
if __name__ == "__main__":
    # Load embeddings
    print("Loading FastText embeddings...")
    fasttext_model = load_fasttext_embeddings(FASTTEXT_FILE)

    # Get distances
    print(f"Calculating distances for related terms of '{TARGET_WORD}'...")
    distances = get_distances_from_target(TARGET_WORD, RELATED_TERMS, fasttext_model)
    for term, distance in distances.items():
        print(f"Similarity between '{TARGET_WORD}' and '{term}': {distance:.4f}")
    labels, vectors = prepare_data(distances, fasttext_model)

    # Apply cPro
    print("Running cPro...")
    projection = run_cpro(vectors)

    # Visualize
    print("Visualizing results...")
    output_file = os.path.join(OUTPUT_DIR, f"{TARGET_WORD}_cpro_related_terms_visualization_fixed.png")
    visualize_cpro_with_similarity(labels, projection, list(distances.values()), output_file)
    print(f"Visualization saved to {output_file}.")
