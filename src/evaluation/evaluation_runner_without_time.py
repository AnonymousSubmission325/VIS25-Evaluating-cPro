import time
import numpy as np
from src.projections.circular_projection import circular_projection
from src.projections.circular_projection_adam import circular_projection_adam
from src.projections.circular_projection_pso import circular_projection_pso
from src.projections.circular_projection_lbfgs import circular_projection_lbfgs
from src.projections.som_projection import som_projection
from src.evaluation.evaluation import run_evaluation
from sklearn.manifold import MDS
from src.data.data_creation import create_multiple_datasets
from src.utils.plotting import plot_original_data, plot_all_projections

# Define colors for target labels, used across all plots
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def run_all_evaluations(show_plots):
    datasets = create_multiple_datasets()

    for dataset_name, sample_data in datasets.items():
        print(f'\nRunning evaluation on {dataset_name} dataset')
        
        start_time = time.time()
        plot_original_data(sample_data, dataset_name, show_plots)
        print(f"Original data plotting completed in {time.time() - start_time:.2f} seconds.")
        
        # Track time for each projection method
        print("Starting projections...")

        # Simulated Annealing-based circular projection
        start_sa = time.time()
        res_cpro_sa = circular_projection(sample_data.drop('target', axis=1))
        print(f"Simulated Annealing cPro completed in {time.time() - start_sa:.2f} seconds.")
        
        # Adam optimizer-based circular projection
        start_adam = time.time()
        res_cpro_adam = circular_projection_adam(sample_data.drop('target', axis=1))
        print(f"Adam cPro completed in {time.time() - start_adam:.2f} seconds.")
        
        # PSO-based circular projection
        start_pso = time.time()
        res_cpro_pso = circular_projection_pso(sample_data.drop('target', axis=1))
        print(f"PSO cPro completed in {time.time() - start_pso:.2f} seconds.")
        
        # L-BFGS-based circular projection
        start_lbfgs = time.time()
        res_cpro_lbfgs = circular_projection_lbfgs(sample_data.drop('target', axis=1))
        print(f"L-BFGS cPro completed in {time.time() - start_lbfgs:.2f} seconds.")
        
        # SOM-based projection
        start_som = time.time()
        res_som = som_projection(sample_data.drop('target', axis=1), show_plots=False, labels=sample_data['target'])
        print(f"SOM Projection completed in {time.time() - start_som:.2f} seconds.")

        # MDS projections
        print("Starting MDS projections...")
        start_mds_2d = time.time()
        mds_2d = MDS(n_components=2, random_state=777, normalized_stress="auto")
        ld_data_mds_2d = mds_2d.fit_transform(sample_data.drop('target', axis=1))
        print(f"MDS 2D completed in {time.time() - start_mds_2d:.2f} seconds.")

        start_mds_1d = time.time()
        mds_1d = MDS(n_components=1, random_state=777, normalized_stress="auto")
        ld_data_mds_1d = mds_1d.fit_transform(sample_data.drop('target', axis=1))
        print(f"MDS 1D completed in {time.time() - start_mds_1d:.2f} seconds.")

        # Prepare all projections for plotting
        projections = [
            (f'{dataset_name} - Simulated Annealing cPro', res_cpro_sa),
            (f'{dataset_name} - Adam cPro', res_cpro_adam),
            (f'{dataset_name} - PSO cPro', res_cpro_pso),
            (f'{dataset_name} - L-BFGS cPro', res_cpro_lbfgs),
            (f'{dataset_name} - SOM', res_som),
            (f'{dataset_name} - MDS 2D', ld_data_mds_2d),
            (f'{dataset_name} - MDS 1D', np.column_stack((ld_data_mds_1d, np.zeros(ld_data_mds_1d.shape[0]))))
        ]

        plot_all_projections(sample_data, projections, show_plots, colors)

        # Prepare evaluation comparisons
        print("Starting evaluation comparisons...")
        other_projections = [
            ('MDS 2D', ld_data_mds_2d),
            ('MDS 1D', np.column_stack((ld_data_mds_1d, np.zeros(ld_data_mds_1d.shape[0])))),
            ('SOM - Circular Projection', np.column_stack((res_som.circle_x, res_som.circle_y))),
            ('Simulated Annealing cPro', np.column_stack((res_cpro_sa.circle_x, res_cpro_sa.circle_y))),
            ('Adam cPro', np.column_stack((res_cpro_adam.circle_x, res_cpro_adam.circle_y))),
            ('PSO cPro', np.column_stack((res_cpro_pso.circle_x, res_cpro_pso.circle_y))),
            ('L-BFGS cPro', np.column_stack((res_cpro_lbfgs.circle_x, res_cpro_lbfgs.circle_y)))
        ]

        start_eval = time.time()
        run_evaluation(sample_data.drop('target', axis=1), sample_data['target'], res_cpro_sa, other_projections)
        print(f"Evaluation completed in {time.time() - start_eval:.2f} seconds.")

if __name__ == '__main__':
    run_all_evaluations(show_plots=True)
