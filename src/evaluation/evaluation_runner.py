# evaluation_runner.py

import time
from src.data.data_creation import create_multiple_datasets
from src.data.data_import import load_datasets, load_artificial_datasets   # Import function to load preprocessed datasets
from src.utils.plotting import plot_original_data, plot_all_projections
from src.utils.data_export import export_evaluation_results
from src.evaluation.evaluation import run_evaluation

# Define colors for target labels, used across all plots
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def run_all_evaluations(show_plots, max_time=300, projections_config=None):
    """
    Runs the complete projection and evaluation on multiple datasets with a time constraint,
    and exports results to 'src/results/records'.

    Parameters:
    -----------
    show_plots : bool
        Whether to show plots or not.
    max_time : int
        Maximum allowed time (in seconds) for each projection method.
    projections_config : list of tuples
        List of projection methods and their configurations.
    """
    if projections_config is None:
        raise ValueError("Projections configuration must be provided.")

    print("[INFO] Starting full evaluation pipeline...")
    print("[INFO] Loading all datasets...")

    print("[INFO] Creating synthetic datasets...")
    generated_datasets = create_multiple_datasets()

    print("[INFO] Importing real-world datasets...")
    imported_datasets = load_datasets()

    print("[INFO] Loading artificial datasets...")
    artificial_datasets = load_artificial_datasets()

    all_datasets = {**generated_datasets, **imported_datasets, **artificial_datasets}
    print(f"[INFO] Total datasets loaded: {len(all_datasets)}")

    all_results = []

    for dataset_name, sample_data in all_datasets.items():
        print("\n" + "-" * 50)
        print(f"Running evaluation on '{dataset_name}' dataset")
        print("-" * 50)

        dataset_metrics = {"Dataset": dataset_name}

        print("[STEP] Plotting original data...")
        start_time = time.time()
        plot_original_data(sample_data, dataset_name, show_plots)
        dataset_metrics["Original Data Plot Time"] = round(time.time() - start_time, 2)
        print(f"[DONE] Original data plotted in {dataset_metrics['Original Data Plot Time']} seconds.")

        projections = []

        for name, func, kwargs in projections_config:
            print(f"[STEP] Running projection: {name}")
            start_time = time.time()
            try:
                if "max_time" in kwargs:
                    kwargs["max_time"] = max_time
                res = func(sample_data.drop('target', axis=1), **kwargs)
                duration = round(time.time() - start_time, 2)
                dataset_metrics[f"{name} Time"] = duration
                print(f"[DONE] {name} completed in {duration} seconds.")
                projections.append((f"{dataset_name} - {name}", res))
            except Exception as e:
                print(f"[ERROR] Projection '{name}' failed: {e}")
                continue

        print("[STEP] Plotting all projections...")
        plot_all_projections(sample_data, projections, show_plots, colors, minimal=False, point_size=50)
        print("[DONE] All projections plotted.")

        print("[STEP] Running evaluation of all projections...")
        start_eval = time.time()
        metrics = run_evaluation(
            sample_data.drop('target', axis=1),
            sample_data['target'],
            projections[0][1],
            projections
        )
        dataset_metrics.update(metrics)
        dataset_metrics["Evaluation Time"] = round(time.time() - start_eval, 2)
        print(f"[DONE] Evaluation completed in {dataset_metrics['Evaluation Time']} seconds.")

        all_results.append(dataset_metrics)

    print("[INFO] Exporting results to disk...")
    export_evaluation_results(all_results, max_time)
    print("[INFO] Evaluation pipeline completed.")
