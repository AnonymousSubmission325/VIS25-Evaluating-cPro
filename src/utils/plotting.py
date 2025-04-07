import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.colors as mcolors
import pandas as pd

# Ensure the directory for saving plots exists
output_dir = "src/results/plots"
os.makedirs(output_dir, exist_ok=True)

def get_consistent_colors(labels):
    """
    Generates a consistent color mapping based on the target labels using a Seaborn color palette.
    """
    unique_labels = np.unique(labels)
    # Use a color palette that differentiates well between classes
    palette = sns.color_palette("husl", len(unique_labels))
    color_mapping = {label: palette[i] for i, label in enumerate(unique_labels)}
    colors = [color_mapping[label] for label in labels]
    return colors

def plot_original_data(df, dataset_name, show_plots, point_size=300):
    """
    Plots the original data in 1D, 2D, or 3D if applicable, or creates a scatterplot matrix for higher dimensions.
    """
    print(df)
    if show_plots:
        feature_columns = df.drop(columns=['target'])
        colors = get_consistent_colors(df['target'])

        if feature_columns.shape[1] == 1:
            plt.figure(figsize=(8, 2))
            plt.scatter(feature_columns.iloc[:, 0], np.zeros_like(feature_columns.iloc[:, 0]), c=colors, edgecolor='k', s=point_size)
            plt.title(f'Original 1D Data: {dataset_name}')
            plt.xlabel('Feature 1')
            plt.yticks([])
            plt.savefig(f"{output_dir}/{dataset_name}_original_1D.png", bbox_inches='tight')
            plt.show()
        elif feature_columns.shape[1] == 2:
            plt.figure(figsize=(8, 8))
            plt.scatter(feature_columns.iloc[:, 0], feature_columns.iloc[:, 1], c=colors, edgecolor='k', s=point_size)
            plt.title(f'Original 2D Data: {dataset_name}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(f"{output_dir}/{dataset_name}_original_2D.png", bbox_inches='tight')
            plt.show()
        elif feature_columns.shape[1] == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(feature_columns.iloc[:, 0], feature_columns.iloc[:, 1], feature_columns.iloc[:, 2], c=colors, edgecolor='k', s=point_size)
            ax.set_title(f'Original 3D Data: {dataset_name}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
            plt.savefig(f"{output_dir}/{dataset_name}_original_3D.png", bbox_inches='tight')
            plt.show()
        elif feature_columns.shape[1] > 3:
            sns.pairplot(df, hue='target', palette=sns.color_palette("husl", len(np.unique(df['target']))))
            plt.suptitle(f'Scatterplot Matrix: {dataset_name}', y=1.02)
            plt.savefig(f"{output_dir}/{dataset_name}_scatterplot_matrix.png", bbox_inches='tight')
            plt.show()
        else:
            print(f'Skipping plot for {dataset_name} as it has an unsupported number of dimensions.')




def plot_all_projections(df, projections, show_plots, base_colors, minimal=False, point_size=20):
    """
    Plots all given projections based on provided coordinates, ensuring colors align with points.
    """
    if show_plots:
        colors = get_consistent_colors(df['target'])[:len(df)]

        for name, res in projections:
            # Default figure size for 2D projections
            fig_size = (8, 8)
            x_stretch_factor = 1  # Default stretch factor for x-axis
            
            if hasattr(res, 'circle_x') and hasattr(res, 'circle_y'):
                x_coords, y_coords = np.array(res.circle_x), np.array(res.circle_y)
            elif isinstance(res, np.ndarray):
                if res.shape[1] == 2:
                    x_coords, y_coords = res[:, 0], res[:, 1]
                    fig_size = (8, 8)  # Standard square size for 2D
                elif res.shape[1] == 1:
                    x_coords, y_coords = res[:, 0], np.zeros_like(res[:, 0])  # Set y to zero for 1D
                    fig_size = (16, 4)  # Wider figure size for 1D
                    x_stretch_factor = 2  # Double the x-axis range for 1D
                else:
                    print(f"Skipping {name}: unsupported projection format.")
                    continue
            elif hasattr(res, 'numpy'):
                x_coords, y_coords = res.numpy()[:, 0], res.numpy()[:, 1]
            else:
                print(f"Skipping {name}: unsupported projection format.")
                continue

            colors = colors[:len(x_coords)]


            # Export projection data
            export_file = os.path.join(output_dir, f"{name.replace(' ', '_')}_projection.csv")
            export_df = pd.DataFrame({
                "x": x_coords,
                "y": y_coords,
                "target": df["target"][:len(x_coords)]
            })
            export_df.to_csv(export_file, index=False)
            print(f"Exported {name} projection to {export_file}")


            # Determine axis limits
            x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
            y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range * x_stretch_factor, y_range) * 1.1  # Apply stretch factor to x-axis range
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            x_min = x_center - max_range / (2 * x_stretch_factor)
            x_max = x_center + max_range / (2 * x_stretch_factor)
            y_min = y_center - max_range / 2
            y_max = y_center + max_range / 2

            plt.figure(figsize=fig_size)
            plt.scatter(x_coords, y_coords, c=colors, edgecolor='white', s=point_size)
            
            if not minimal:
                plt.title(f'{name} Projection')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)
                plt.gca().set_aspect('auto' if x_stretch_factor > 1 else 'equal', adjustable='box')
            else:
                plt.axis('off')

            plt.savefig(f"{output_dir}/{name.replace(' ', '_')}_projection{'_minimal' if minimal else ''}.png", bbox_inches='tight')
            plt.show()

    
def export_projection_data(df, projections, output_dir="src/results/projections"):
    os.makedirs(output_dir, exist_ok=True)
    target_labels = df['target']

    for name, res in projections:
        # Extract x, y coordinates
        if hasattr(res, 'circle_x') and hasattr(res, 'circle_y'):
            x_coords, y_coords = np.array(res.circle_x), np.array(res.circle_y)
        elif isinstance(res, np.ndarray):
            if res.shape[1] == 2:
                x_coords, y_coords = res[:, 0], res[:, 1]
            elif res.shape[1] == 1:
                x_coords, y_coords = res[:, 0], np.zeros_like(res[:, 0])
            else:
                print(f"Skipping {name}: unsupported projection format.")
                continue
        elif hasattr(res, 'numpy'):
            x_coords, y_coords = res.numpy()[:, 0], res.numpy()[:, 1]
        else:
            print(f"Skipping {name}: unsupported projection format.")
            continue

        # Create a DataFrame with x, y, and target class
        export_df = pd.DataFrame({
            "x": x_coords,
            "y": y_coords,
            "target": target_labels[:len(x_coords)]
        })

        # Save to CSV
        file_path = os.path.join(output_dir, f"{name.replace(' ', '_')}_projection_data.csv")
        export_df.to_csv(file_path, index=False)
        print(f"Exported {name} projection data to {file_path}")
