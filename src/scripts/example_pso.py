import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from src.projections.circular_projection_pso import circular_projection_pso  # Import the PSO version


def create_sample_data():
    # Set parameters
    n_points = 200  # Total number of data points
    n_clusters = 4  # Number of clusters

    # Generate 3D data with 4 blobs
    data, labels = make_blobs(n_samples=n_points, centers=n_clusters, n_features=3, random_state=777)

    # Convert the generated data to a pandas DataFrame
    sample_data = pd.DataFrame(data, columns=['x', 'y', 'z'])

    # Add the target variable to the DataFrame
    sample_data['target'] = labels

    return sample_data


def plot_circular_projection(df, res):
    # Define a custom color palette (Blue, Orange, Green, Red)
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Map target values to colors
    colors = [palette[i] for i in df['target']]

    # Create the plot
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.title('Circular Projection of the Data Points (PSO)')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(res.circle_x, res.circle_y, c=colors, edgecolor='white', s=35)


def plot_loss(res):
    plt.subplots()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('PSO Optimization: Loss Reduction Over Iterations')
    plt.plot(res.loss_records)


def plot_distances(res):
    plt.subplots()
    plt.xlabel('High-dimensional distances')
    plt.ylabel('Low-dimensional distances')
    plt.title('High-dimensional vs. Low-dimensional Distances (PSO)')
    plt.scatter(res.hd_dist_matrix.flatten(), res.ld_dist_matrix.flatten(), alpha=0.5, s=5)


if __name__ == '__main__':

    plt.style.use('ggplot')
    sample_data = create_sample_data()

    # Call the PSO-based circular projection
    res = circular_projection_pso(sample_data[['x', 'y', 'z']])

    plot_circular_projection(sample_data, res)
    plot_loss(res)
    plot_distances(res)

    plt.show()
