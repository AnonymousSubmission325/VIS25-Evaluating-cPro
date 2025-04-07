import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from src.projections.circular_projection import circular_projection
from src.evaluation.evaluation import run_evaluation
from sklearn.manifold import MDS


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
    plt.title('Circular Projection of the Data Points')
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(res.circle_x, res.circle_y, c=colors, edgecolor='white', s=35)


def plot_loss(res):
    plt.subplots()
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Dual Annealing Optimization: Loss Reduction Over Iterations')
    plt.plot(res.loss_records)


def plot_distances(res):
    plt.subplots()
    plt.xlabel('High-dimensional distances')
    plt.ylabel('Low-dimensional distances')
    plt.title('High-dimensional vs. Low-dimensional Distances')
    plt.scatter(res.hd_dist_matrix.flatten(), res.ld_dist_matrix.flatten(), alpha=0.5, s=5)


if __name__ == '__main__':
    plt.style.use('ggplot')
    sample_data = create_sample_data()

    # Run circular projection
    res = circular_projection(sample_data[['x', 'y', 'z']])

    # Plot results for circular projection
    plot_circular_projection(sample_data, res)
    plot_loss(res)
    plot_distances(res)
    plt.show()

    # Prepare other dimensionality reduction techniques
    print('\nRun evaluation comparing to multiple MDS methods')

    # 2D MDS
    mds_2d = MDS(n_components=2, random_state=777)
    ld_data_mds_2d = mds_2d.fit_transform(sample_data[['x', 'y', 'z']])

    # 1D MDS
    mds_1d = MDS(n_components=1, random_state=777)
    ld_data_mds_1d = mds_1d.fit_transform(sample_data[['x', 'y', 'z']])

    # Run evaluation comparing cPro with MDS 2D and MDS 1D
    other_projections = [
        ('MDS 2D', ld_data_mds_2d),
        ('MDS 1D', ld_data_mds_1d)
    ]

    run_evaluation(sample_data[['x', 'y', 'z']], sample_data['target'], res, other_projections)
